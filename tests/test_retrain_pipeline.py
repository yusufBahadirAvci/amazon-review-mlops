"""Tests for Ceylin's auto-retrain pipeline: monitoring/retrain_pipeline.py.

The real scripts, MLflow and the API are replaced by fakes, so no server,
database or network is needed.
"""

import json

import pytest
import requests

from monitoring import retrain_pipeline as rp

DRIFT = "monitoring/drift_report.py"
VALIDATE = "data/validate_data.py"
TRAIN = "training/train.py"


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Replace the scripts, MLflow and the API with fakes we can control."""
    monkeypatch.setattr(rp, "REPORTS_DIR", tmp_path)
    state = {
        "drift": True,  # True / False, or None = drift script writes no report
        "codes": {},  # exit code per script, default 0
        "versions": ["3", "3"],  # production version before / after training
        "reload_fails": False,
        "steps": [],
        "reloads": [],
    }

    def fake_run_step(name, args):
        script = args[0]
        state["steps"].append(script)
        if script == DRIFT and state["drift"] is not None:
            report = tmp_path / "drift_db_20991231_235959.json"
            report.write_text(json.dumps({"drift_detected": state["drift"]}))
        return state["codes"].get(script, 0)

    def fake_production_version(client):
        return state["versions"].pop(0)

    def fake_reload(api_url):
        state["reloads"].append(api_url)
        if state["reload_fails"]:
            raise requests.ConnectionError("API is down")
        return {"model_version": "4"}

    monkeypatch.setattr(rp, "run_step", fake_run_step)
    monkeypatch.setattr(rp, "production_version", fake_production_version)
    monkeypatch.setattr(rp, "reload_api", fake_reload)
    return state


def run(force=False):
    return rp.run_pipeline(source="db", trials=3, force=force, api_url="http://api", client=None)


def test_no_drift_stops_after_drift_check(fake_pipeline):
    fake_pipeline["drift"] = False
    assert run() == rp.EXIT_OK
    assert fake_pipeline["steps"] == [DRIFT]
    assert fake_pipeline["reloads"] == []


def test_no_report_means_nothing_to_do(fake_pipeline):
    fake_pipeline["drift"] = None  # e.g. fewer than 50 crawled reviews
    assert run() == rp.EXIT_OK
    assert fake_pipeline["steps"] == [DRIFT]


def test_failed_drift_check_is_a_step_failure(fake_pipeline):
    fake_pipeline["codes"] = {DRIFT: 1}
    assert run() == rp.EXIT_STEP_FAILED
    assert fake_pipeline["steps"] == [DRIFT]


def test_invalid_data_stops_before_training(fake_pipeline):
    fake_pipeline["codes"] = {VALIDATE: 1}
    assert run() == rp.EXIT_INVALID_DATA
    assert fake_pipeline["steps"] == [DRIFT, VALIDATE]


def test_new_production_model_triggers_api_reload(fake_pipeline):
    fake_pipeline["versions"] = ["3", "4"]
    assert run() == rp.EXIT_OK
    assert fake_pipeline["steps"] == [DRIFT, VALIDATE, TRAIN]
    assert fake_pipeline["reloads"] == ["http://api"]


def test_unchanged_production_model_skips_api_reload(fake_pipeline):
    fake_pipeline["versions"] = ["3", "3"]
    assert run() == rp.EXIT_OK
    assert fake_pipeline["steps"] == [DRIFT, VALIDATE, TRAIN]
    assert fake_pipeline["reloads"] == []


def test_api_reload_failure_is_reported(fake_pipeline):
    fake_pipeline["versions"] = ["3", "4"]
    fake_pipeline["reload_fails"] = True
    assert run() == rp.EXIT_STEP_FAILED


def test_force_skips_drift_check(fake_pipeline):
    fake_pipeline["versions"] = ["3", "4"]
    assert run(force=True) == rp.EXIT_OK
    assert fake_pipeline["steps"] == [VALIDATE, TRAIN]


def test_latest_report_ignores_old_files(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "REPORTS_DIR", tmp_path)
    report = tmp_path / "drift_db_old.json"
    report.write_text("{}")
    future = report.stat().st_mtime + 100
    assert rp.latest_report("db", since=future) is None
    assert rp.latest_report("db", since=0) == report
