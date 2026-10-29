"""Data drift report with Evidently.

Compares the training data (reference) with new reviews (current) on:
- text_length, word_count  (how people write)
- rating                   (star distribution)
- prediction               (what the production model predicts)

Current data sources:
    --source db      reviews collected by the crawler (PostgreSQL "reviews" table)
    --source sample  reviews from Amazon categories NOT used in training
                     (for testing the report before the crawler has enough data)

Output:
- monitoring/reports/drift_<source>_<time>.html  (open in a browser)
- monitoring/reports/drift_<source>_<time>.json  (summary for the retrain loop)
- the HTML report and the drift share are also logged to MLflow
  (experiment "drift-monitoring")

Usage:
    python monitoring/drift_report.py --source sample
    python monitoring/drift_report.py --source db
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from huggingface_hub import HfFileSystem
from sqlalchemy import create_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_reviews.csv"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
MODEL_URI = "models:/sentiment-model@production"
EXPERIMENT_NAME = "drift-monitoring"
HF_REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
SAMPLE_CATEGORIES = ["Video_Games", "Pet_Supplies"]  # not used in training
SAMPLE_LINES_PER_CATEGORY = 1_000
REFERENCE_SAMPLE_SIZE = 5_000
MIN_CURRENT_ROWS = 50
DRIFT_SHARE_THRESHOLD = 0.5  # dataset drift if half of the columns drifted
RANDOM_SEED = 42


def load_reference() -> pd.DataFrame:
    """A random sample of the training data."""
    df = pd.read_csv(REFERENCE_PATH, keep_default_na=False)
    n = min(len(df), REFERENCE_SAMPLE_SIZE)
    return df.sample(n=n, random_state=RANDOM_SEED)[["text", "rating"]]


def load_current_from_db() -> pd.DataFrame:
    """Reviews written by the crawler into PostgreSQL."""
    load_dotenv(PROJECT_ROOT / ".env")
    url = (
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"
    )
    return pd.read_sql("SELECT text, rating FROM reviews", create_engine(url))


def load_current_from_sample() -> pd.DataFrame:
    """Reviews from categories the model has never seen (streamed, not downloaded)."""
    fs = HfFileSystem()
    rows = []
    for category in SAMPLE_CATEGORIES:
        path = f"datasets/{HF_REPO_ID}/raw/review_categories/{category}.jsonl"
        print(f"Reading '{category}'...")
        with fs.open(path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                if line_number >= SAMPLE_LINES_PER_CATEGORY:
                    break
                record = json.loads(line)
                rows.append({"text": record.get("text"), "rating": record.get("rating")})
    return pd.DataFrame(rows)


def add_features(df: pd.DataFrame, model) -> pd.DataFrame:
    """Clean the rows and compute the columns Evidently compares."""
    df = df.dropna(subset=["text", "rating"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["rating"] = df["rating"].astype(int).astype(str)
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["prediction"] = model.predict(df[["text"]])
    return df[["text_length", "word_count", "rating", "prediction"]]


def metric_label(metric: dict) -> str:
    """Readable name of an Evidently metric (the key differs between versions)."""
    for key in ("metric_name", "metric_id", "name"):
        if metric.get(key):
            return str(metric[key])
    return "metric"


def drifted_share(result: dict) -> float | None:
    """Share of drifted columns: the metric whose value has 'count' and 'share'."""
    for metric in result.get("metrics", []):
        value = metric.get("value")
        if isinstance(value, dict) and "share" in value:
            return float(value["share"])
    return None


def save_outputs(snapshot, summary: dict, source: str) -> Path:
    """Write the HTML report and the JSON summary (reports are not committed)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    gitignore = REPORTS_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = REPORTS_DIR / f"drift_{source}_{stamp}.html"
    snapshot.save_html(str(html_path))
    html_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["db", "sample"], default="sample")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(MODEL_URI)

    reference = add_features(load_reference(), model)
    raw_current = load_current_from_db() if args.source == "db" else load_current_from_sample()
    current = add_features(raw_current, model)
    print(f"Reference rows: {len(reference)}  Current rows: {len(current)}")

    if len(current) < MIN_CURRENT_ROWS:
        print(
            f"Only {len(current)} current rows (need at least {MIN_CURRENT_ROWS}). "
            "Collect more reviews or use --source sample."
        )
        return

    definition = DataDefinition(
        numerical_columns=["text_length", "word_count"],
        categorical_columns=["rating", "prediction"],
    )
    snapshot = Report(metrics=[DataDriftPreset()]).run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
    )

    result = snapshot.dict()
    for metric in result.get("metrics", []):
        print(f"  {metric_label(metric)}: {metric.get('value')}")

    share = drifted_share(result)
    drift_detected = share is not None and share >= DRIFT_SHARE_THRESHOLD
    summary = {
        "source": args.source,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drifted_columns_share": share,
        "drift_detected": drift_detected,
        "prediction_share_reference": reference["prediction"]
        .value_counts(normalize=True)
        .round(3)
        .to_dict(),
        "prediction_share_current": current["prediction"]
        .value_counts(normalize=True)
        .round(3)
        .to_dict(),
    }
    html_path = save_outputs(snapshot, summary, args.source)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"drift-{args.source}"):
        mlflow.log_params({"source": args.source, "current_rows": len(current)})
        if share is not None:
            mlflow.log_metric("drifted_columns_share", share)
        mlflow.log_metric("drift_detected", int(drift_detected))
        mlflow.log_artifact(str(html_path))

    print(f"\nDrifted columns share: {share}")
    print(f"Drift detected: {drift_detected}")
    print(f"Prediction share (training): {summary['prediction_share_reference']}")
    print(f"Prediction share (current):  {summary['prediction_share_current']}")
    print(f"Report: {html_path}")


if __name__ == "__main__":
    main()
