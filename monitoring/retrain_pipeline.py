"""Auto-retrain pipeline: drift check -> data validation -> training -> API reload.

Steps:
1. Run monitoring/drift_report.py and read the JSON report it just wrote.
   Stop if there is no new report (too few rows) or no drift.
   (--force skips this step.)
2. Run data/validate_data.py. Stop if the data is broken.
3. Run training/train.py. Its promotion rule decides whether the
   "production" alias moves to the new model version.
4. Only if the production version changed: call the API endpoint
   POST {API_URL}/api/model/reload so the API loads the new model.

Exit codes (for the scheduler that will run this script):
    0  finished (retrained or nothing to do)
    1  data validation failed
    2  a step failed (drift report, training or API reload)

Usage:
    python monitoring/retrain_pipeline.py                  # crawled reviews (db)
    python monitoring/retrain_pipeline.py --source sample  # test data
    python monitoring/retrain_pipeline.py --force          # skip the drift check

Environment variables:
    MLFLOW_TRACKING_URI  default http://localhost:5000
    API_URL              default http://localhost:8000
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import requests
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
MODEL_NAME = "sentiment-model"
PRODUCTION_ALIAS = "production"
RELOAD_PATH = "/api/model/reload"
REPORT_TIME_MARGIN_SECONDS = 2  # file timestamps on Windows can be a bit coarse

EXIT_OK = 0
EXIT_INVALID_DATA = 1
EXIT_STEP_FAILED = 2


def run_step(name: str, args: list[str]) -> int:
    """Run one project script with the current Python and return its exit code."""
    print(f"\n=== {name}: python {' '.join(args)} ===", flush=True)
    return subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT).returncode


def latest_report(source: str, since: float) -> Path | None:
    """Newest drift JSON for this source written after `since` (or None)."""
    reports = [
        path for path in REPORTS_DIR.glob(f"drift_{source}_*.json") if path.stat().st_mtime >= since
    ]
    return max(reports, key=lambda path: path.stat().st_mtime, default=None)


def production_version(client: MlflowClient) -> str | None:
    """Version number behind the production alias, or None if there is none."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except MlflowException:
        return None
    return str(version.version)


def reload_api(api_url: str) -> dict:
    """Ask the API to load the current production model."""
    response = requests.post(api_url.rstrip("/") + RELOAD_PATH, timeout=60)
    response.raise_for_status()
    return response.json()


def run_pipeline(source: str, trials: int, force: bool, api_url: str, client: MlflowClient) -> int:
    """Run all steps and return an exit code (see module docstring)."""
    started = time.time()

    if force:
        print("--force: skipping the drift check.")
    else:
        if run_step("Drift check", ["monitoring/drift_report.py", "--source", source]):
            print("Drift check failed.")
            return EXIT_STEP_FAILED
        report = latest_report(source, started - REPORT_TIME_MARGIN_SECONDS)
        if report is None:
            print("No new drift report (probably too few rows). Nothing to do.")
            return EXIT_OK
        summary = json.loads(report.read_text())
        if not summary.get("drift_detected"):
            print(f"No drift ({report.name}). Nothing to do.")
            return EXIT_OK
        print(f"Drift detected ({report.name}).")

    if run_step("Data validation", ["data/validate_data.py"]):
        print("Data validation failed. Training stopped.")
        return EXIT_INVALID_DATA

    before = production_version(client)
    if run_step("Training", ["training/train.py", "--trials", str(trials)]):
        print("Training failed.")
        return EXIT_STEP_FAILED
    after = production_version(client)

    if after == before:
        print(f"Production model unchanged (version {after}). No API reload needed.")
        return EXIT_OK

    print(f"Production model changed: {before} -> {after}. Reloading the API...")
    try:
        result = reload_api(api_url)
    except requests.RequestException as error:
        print(f"API reload failed: {error}")
        return EXIT_STEP_FAILED
    print(f"API reloaded: {result}")
    return EXIT_OK


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["db", "sample"], default="db")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    api_url = os.getenv("API_URL", "http://localhost:8000")
    sys.exit(run_pipeline(args.source, args.trials, args.force, api_url, MlflowClient()))


if __name__ == "__main__":
    main()
