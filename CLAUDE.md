# CLAUDE.md

Rules for Claude Code in this repository. Two people work here: Bahadır and Ceylin.

## 1. Read first
Always read `PROJECT_CONTEXT.md` before starting. Section 5 is the contract between Bahadır and Ceylin; section 7 is the current status.

## 2. Folder ownership
- **Bahadır:** `app/`, `tests/`, `Dockerfile`, `.dockerignore`, `.github/workflows/`, `prometheus.yml`
- **Ceylin:** `data/`, `training/`, `monitoring/`, `.dvc/`, `data/raw/*.dvc`
- **Shared** (announce to the other person before editing): `requirements.txt`, `docker-compose.yml`, `.gitignore`, `.pre-commit-config.yaml`, `PROJECT_CONTEXT.md`, `pyproject.toml`, `CLAUDE.md`

Never edit the other person's folders.

## 3. Git
- Never push to `main` directly.
- Start new work from an up-to-date `main`.
- Before opening a PR, merge `main` into the branch (no rebase).
- Merge PRs with "Create a merge commit".
- Never force-push.

## 4. requirements.txt
- Only add packages in your own section.
- Do not reorder lines.
- Run `pip check` after installing (mlflow 2.11.1 is old and conflicts easily).

## 5. docker-compose.yml
- Add new services as new blocks.
- Do not edit the `mlflow-server` or `postgres-db` blocks, the volume names or the network name.

## 6. Docker
Never run `docker compose down -v` (it deletes the database and MLflow volumes).

## 7. Tests
Tests must not need a running MLflow, Postgres or internet connection. Mock the model.

## 8. Data and MLflow
- Read the training CSV with `pd.read_csv(path, keep_default_na=False)` (otherwise texts like "NA" become empty values).
- Never use the MLflow "Default" experiment.
