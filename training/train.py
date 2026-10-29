"""Train the sentiment model with Optuna and log it to MLflow.

Steps:
1. Load data/raw/amazon_reviews.csv and split it 70 / 15 / 15
   (train / validation / test), keeping the class balance.
2. Optuna tries different settings (classifier type, C, n-grams, ...)
   and keeps the one with the best macro F1 on the validation set.
   Every trial is logged to MLflow as a nested run.
3. The best settings are retrained on train + validation and scored
   on the untouched test set.
4. The model is registered as "sentiment-model". It gets the
   "production" alias only if it beats the current production model.

Project contract: the model takes a DataFrame with a "text" column and
returns the label string (negative / neutral / positive). TF-IDF lives
inside the model, so the API needs no extra preprocessing code.

Usage:
    python training/train.py             # 20 Optuna trials
    python training/train.py --trials 3  # quick test run
"""

import argparse
import os
from pathlib import Path

import mlflow
import optuna
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_reviews.csv"
DVC_FILE = DATA_PATH.parent / (DATA_PATH.name + ".dvc")
EXPERIMENT_NAME = "sentiment-training"
MODEL_NAME = "sentiment-model"
PRODUCTION_ALIAS = "production"
RANDOM_SEED = 42
DEFAULT_TRIALS = 20


def load_data():
    """Load the dataset and split it into train / validation / test."""
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    df = df[df["text"].str.strip() != ""]
    x, y = df[["text"]], df["label"]

    x_train, x_rest, y_train, y_rest = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=RANDOM_SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_SEED
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def data_version() -> str:
    """Read the DVC md5 of the dataset, so each run records which data it used."""
    for line in DVC_FILE.read_text().splitlines():
        if "md5:" in line:
            return line.split("md5:")[1].strip()
    return "unknown"


def build_pipeline(params: dict) -> Pipeline:
    """TF-IDF on the "text" column, followed by a linear classifier."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, params["ngram_max"]),
        min_df=params["min_df"],
        max_features=params["max_features"],
        sublinear_tf=params["sublinear_tf"],
    )
    if params["classifier"] == "logreg":
        classifier = LogisticRegression(C=params["C"], max_iter=1000)
    else:
        classifier = LinearSVC(C=params["C"], max_iter=5000)

    features = ColumnTransformer([("tfidf", vectorizer, "text")])
    return Pipeline([("features", features), ("classifier", classifier)])


def suggest_params(trial: optuna.Trial) -> dict:
    """The settings Optuna is allowed to try."""
    return {
        "classifier": trial.suggest_categorical("classifier", ["logreg", "linearsvc"]),
        "C": trial.suggest_float("C", 0.01, 10.0, log=True),
        "ngram_max": trial.suggest_int("ngram_max", 1, 2),
        "min_df": trial.suggest_int("min_df", 1, 5),
        "max_features": trial.suggest_categorical("max_features", [20_000, 50_000, 100_000]),
        "sublinear_tf": trial.suggest_categorical("sublinear_tf", [True, False]),
    }


def tune(x_train, y_train, x_val, y_val, n_trials: int) -> optuna.Study:
    """Run the Optuna search; each trial becomes a nested MLflow run."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            model = build_pipeline(params).fit(x_train, y_train)
            score = f1_score(y_val, model.predict(x_val), average="macro")
            mlflow.log_params(params)
            mlflow.log_metric("val_f1_macro", score)
        print(f"  Trial {trial.number}: val macro F1 = {score:.4f}")
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def production_score(client: MlflowClient) -> float | None:
    """Test macro F1 of the current production model, or None if there is none."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except MlflowException:
        return None
    return client.get_run(version.run_id).data.metrics.get("test_f1_macro")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    x_train, x_val, x_test, y_train, y_val, y_test = load_data()
    print(f"Train: {len(x_train)}  Validation: {len(x_val)}  Test: {len(x_test)}")
    print(f"Running {args.trials} Optuna trials (MLflow: {tracking_uri})")

    with mlflow.start_run(run_name="optuna-search"):
        study = tune(x_train, y_train, x_val, y_val, args.trials)
        best_params = study.best_params
        print(f"\nBest validation macro F1: {study.best_value:.4f}")
        print(f"Best settings: {best_params}")

        x_full = pd.concat([x_train, x_val])
        y_full = pd.concat([y_train, y_val])
        model = build_pipeline(best_params).fit(x_full, y_full)
        predictions = model.predict(x_test)

        test_f1 = f1_score(y_test, predictions, average="macro")
        test_accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        print(f"\nTest macro F1: {test_f1:.4f}  Test accuracy: {test_accuracy:.4f}")
        print(report)

        mlflow.log_params(best_params)
        mlflow.log_params(
            {
                "n_trials": args.trials,
                "train_rows": len(x_full),
                "data_md5": data_version(),
            }
        )
        mlflow.log_metrics(
            {
                "best_val_f1_macro": study.best_value,
                "test_f1_macro": test_f1,
                "test_accuracy": test_accuracy,
            }
        )
        mlflow.log_text(report, "classification_report.txt")

        sample = x_test.head(5)
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(sample, model.predict(sample)),
            input_example=x_test.head(3),
        )

    registered = mlflow.register_model(model_info.model_uri, MODEL_NAME)
    client = MlflowClient()
    current = production_score(client)
    if current is None or test_f1 > current:
        client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, registered.version)
        print(f"Version {registered.version} is now '{PRODUCTION_ALIAS}'.")
    else:
        print(
            f"Version {registered.version} registered, but production stays "
            f"(production F1 {current:.4f} >= new F1 {test_f1:.4f})."
        )


if __name__ == "__main__":
    main()
