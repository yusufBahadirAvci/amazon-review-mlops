# PROJECT CONTEXT — read fully before helping

Two students on a university MLOps project: Ceylin (data and model) and Bahadır (application and infrastructure). We are beginners: little Java knowledge, new to Python. Explain simply, work in small steps, and warn early if something will cause problems later.

## 1. Goal
End-to-end MLOps system that classifies Amazon product reviews by sentiment (negative / neutral / positive).
- Training data: Hugging Face "McAuley-Lab/Amazon-Reviews-2023" (English, 1-5 star ratings).
- Live data: web crawler collecting English reviews from English-language Amazon sites only (amazon.com, amazon.co.uk, amazon.ca, amazon.com.au, amazon.in). Used for drift monitoring, not for initial training.
- No Turkish anywhere in the project (no Turkish training data, no Turkish marketplaces, no Turkish test fixtures).
No tool or language restrictions from the professor; the result is what matters.

## 2. Decisions already made
- Language: Python 3.12.10 (the old Java version is kept on branch "java-backup", do not use it).
- IDE: PyCharm, plus Claude Code inside the IDE. Team rules for Claude Code are in CLAUDE.md.
- Database: PostgreSQL 15 in Docker. MLflow server v2.22.4 in Docker (port 5000). Both in docker-compose.yml, together with the "api" service.
- MLflow client is pinned to the same version as the server: mlflow==2.22.4.
- pandas is pinned to 2.2.3 because mlflow 2.x does not support pandas 3.
- Secrets: .env (never committed); .env.example shows the keys. Default DB port 5432 (Bahadır uses 5433 in his own .env).
- Code quality: pre-commit with ruff (lint + format) and pytest on every commit. GitHub Actions CI runs the same checks plus a Docker build on every PR.
- Web API: FastAPI.
- Crawler: requests + BeautifulSoup, live mode + fallback mode parsing a saved HTML page (we do NOT bypass Amazon's bot blocking).
- Data versioning: DVC with a DagsHub remote.
- Never run "docker compose down -v" (it deletes the database and MLflow volumes). Use "docker compose up -d <service>" or "docker compose stop".

## 3. Architecture (5 stages)
1. SETUP: GitHub repo, Docker Compose, pre-commit, Python skeleton. (done)
2. DATA: DVC dataset versioning, Great Expectations validation, crawler storing reviews in PostgreSQL. (done)
3. MODEL: sentiment model with Optuna tuning, logged to MLflow Tracking and Model Registry. (done)
4. DEPLOY: FastAPI /api/predict loading the model from MLflow, multi-stage Docker build, CI with GitHub Actions. (done)
   Note: images are pushed to GitHub Container Registry (ghcr.io); no Docker Hub account is needed.
5. MONITOR: Evidently drift report (done), auto-retrain pipeline (done, needs a scheduler), Prometheus metrics (not started).

## 4. Roles and folder ownership
| Owner | Folders / files |
|---|---|
| Bahadır | app/, Dockerfile, .dockerignore, requirements-api.txt, .github/workflows/, Prometheus config |
| Ceylin | data/, training/, monitoring/, .dvc/, data/raw/*.dvc |
| Shared (tell the other person before changing) | requirements.txt, docker-compose.yml, .gitignore, .pre-commit-config.yaml, pyproject.toml, PROJECT_CONTEXT.md, CLAUDE.md, tests/conftest.py |

- tests/: everyone writes and changes only their own test files.
  - Ceylin: test_data_pipeline.py, test_training.py, test_drift.py, test_retrain_pipeline.py
  - Bahadır: all other test files and fixtures
- requirements.txt: each person adds packages only in their own section; do not reorder or reformat other sections.
- docker-compose.yml: add new services as new blocks; do not rename the existing volumes (postgres_data, mlflow_data) or the network (mlops-network).
- Together: auto-retrain trigger, end-to-end test, README, report, presentation.

## 5. Contract between our parts (do not change without telling the other person)
- Labels from stars: 1-2 = negative, 3 = neutral, 4-5 = positive.
- MLflow registered model name: "sentiment-model". The model ready for use has the alias "production".
- The API loads it with mlflow.pyfunc.load_model("models:/sentiment-model@production").
  - Input: pandas DataFrame with a "text" column of raw review text.
  - Output: label strings.
  - Preprocessing lives inside the model (scikit-learn Pipeline: TF-IDF + LogisticRegression or LinearSVC). No probabilities (LinearSVC has no predict_proba).
- MLflow tracking URI from the env variable MLFLOW_TRACKING_URI: http://localhost:5000 on the host, http://mlflow-server:5000 inside Docker.
- Crawled reviews: PostgreSQL table "reviews" with columns including text, rating, source, marketplace, crawled_at. The drift analysis reads text and rating from this table.
- API endpoints used by the ML side: POST /api/model/reload loads the current production model (called by the auto-retrain pipeline; a failed reload keeps the old model).
- The model's training environment and the API image must use the same versions (mlflow 2.22.4, scikit-learn 1.9.1, pandas 2.2.3). requirements-api.txt takes its versions from requirements.txt with "-c requirements.txt".

## 6. Git workflow
- Never push directly to main. Each task gets a new branch from an up-to-date main; merge through Pull Requests with "Create a merge commit" (not squash, not rebase).
- Start every session with: git switch main, git pull, pip install -r requirements.txt.
- Before opening a PR: git pull --no-rebase origin main into your branch, fix conflicts locally, run pytest, then push.
- After a PR is merged, never commit to that branch again; start a new branch.
- Shared files: one small PR per change when possible, merged quickly.
- If a merge conflict happens in PROJECT_CONTEXT.md, keep both people's lines and only remove the conflict markers.

## 7. Current status (last updated: 2026-10-29)

### Data (Ceylin)
- data/download_data.py builds data/raw/amazon_reviews.csv: 30,000 English reviews from 10 product categories (All_Beauty, Amazon_Fashion, Appliances, Baby_Products, Cell_Phones_and_Accessories, Electronics, Home_and_Kitchen, Office_Products, Sports_and_Outdoors, Toys_and_Games), 1,000 per class per category, sampled from the first 100k reviews of each category, RANDOM_SEED=42. Columns: text, rating, label, category, source.
- Texts like "n/a", "null", "none" are removed (pandas reads "n/a" as NaN). Always read the CSV with keep_default_na=False.
- Tracked with DVC. Current md5: 8bbbae6b4e0f486fbe407e0a8a20d1fc.
- DVC remote: DagsHub repo "ceylinkucuk/amazon-review-mlops-data", remote name "dagshub" (default), URL https://dagshub.com/ceylinkucuk/amazon-review-mlops-data.dvc. Credentials are set per person with "dvc remote modify dagshub --local ..." (stored in .dvc/config.local, never committed). To get the data: be added as a collaborator on DagsHub, set your own token locally, then "dvc pull".
- data/validate_data.py (Great Expectations 1.24.1): 10 checks (columns exist and are not null, text not empty, rating 1-5, label in the 3 classes, at least 1000 rows). Exit code 0 = pass, 1 = fail. validate_reviews(df) is reusable.
- Dropped datasets: Cleanlab/amazon-reviews (no star ratings, no neutral class) and a Turkish dataset.

### Model (Ceylin)
- training/train.py: 70/15/15 stratified split (seed 42), TF-IDF + LogisticRegression/LinearSVC, 20 Optuna trials (--trials N). Experiment "sentiment-training" with nested trial runs. Logs params, data_md5, metrics and classification_report.txt; registers "sentiment-model".
- Promotion rule: the "production" alias moves only if the new test_f1_macro is strictly greater than the current production's.
- Current production: version 3, trained with MLflow 2.22.4. Test macro F1 0.7001, accuracy 0.70. Per class F1: negative 0.69, neutral 0.59 (weakest), positive 0.82.
- The "Default" MLflow experiment is never used (it has an old artifact location).

### Monitoring (Ceylin)
- monitoring/drift_report.py (Evidently 0.7.23) compares the training data with current reviews on text_length, word_count, rating and model prediction.
  - --source db reads the "reviews" table (needs at least 50 rows).
  - --source sample uses categories not used in training (Video_Games, Pet_Supplies).
  - drift_detected uses only the trigger columns text_length and word_count. rating and prediction are information only, because the balanced training data always differs from real reviews (~66% positive).
  - Writes HTML + JSON to monitoring/reports/ (gitignored) and logs to the MLflow experiment "drift-monitoring".
- monitoring/retrain_pipeline.py: drift_report.py -> newest JSON -> if drift_detected: validate_data.py (stop on exit 1) -> train.py -> if the production version changed: POST {API_URL}/api/model/reload (API_URL default http://localhost:8000).
  - Options: --source db|sample, --trials N, --force (skip the drift check).
  - Exit codes: 0 ok or nothing to do, 1 invalid data, 2 a step failed.
  - Real run (--source sample --trials 3): drift -> validation -> version 4 (F1 0.6795) not promoted, production stayed 3, no API reload.

### Application and infrastructure (Bahadır)
- Python skeleton: FastAPI /health, routes, DB models; crawler (parser, rating, service) with tests on saved Amazon / captcha pages.
- No Turkish: the crawler accepts only English marketplaces (amazon.com, amazon.co.uk, amazon.ca, amazon.com.au, amazon.in; others get 400), parses ratings from the star CSS class or the English "x out of 5 stars" label only, and skips translated reviews and "reviews from other countries" with a warning (counter skipped_foreign). Turkish test fixture removed.
- POST /api/predict ({"texts": [...]}, 1-100 texts, max 5000 chars each -> {"predictions": [...], "model_version": "..."}) loads models:/sentiment-model@production once at startup (MLFLOW_TRACKING_URI, MODEL_URI settings). POST /api/model/reload reloads it for auto-retrain (a failed reload keeps the old model). /health shows "model" and "model_version". If MLflow is down the app still starts and /predict returns 503.
- Docker + CI:
  - Multi-stage Dockerfile (python:3.12-slim, non-root, uvicorn on 8000) installs requirements-api.txt with "-c requirements.txt", so all versions come from requirements.txt.
  - "api" service in docker-compose.yml: talks to postgres-db:5432 and http://mlflow-server:5000 inside Docker, waits up to 60 s for MLflow at startup, healthcheck on /health. Start with "docker compose up -d --build" -> http://localhost:8000.
  - GitHub Actions (.github/workflows/ci.yml) on PRs and pushes to main: ruff, ruff format --check, pytest, docker build. On main it also pushes ghcr.io/yusufbahadiravci/amazon-review-mlops.

### MLflow upgrade (Ceylin, PR #10, merged)
- Server image and client upgraded from 2.11.1 to 2.22.4 (v2.22.5 has no Docker image). Tested first in a separate venv with a temporary server; CI green.
- The workaround pins needed by 2.11.1 were removed (setuptools, protobuf, opentelemetry-proto, virtualenv); pytest is back to 9.2.0.
- Upgrading an existing environment:
  1. Back up the mlflow_data volume.
  2. Stop the server, pull the new image, run "docker compose run --rm mlflow-server mlflow db upgrade sqlite:///mlflow.db", start the server again.
  3. Retrain, then move the production alias to the new version once by hand (same data and seed give the same F1, so the promotion rule does not move it).

### Not started
- Prometheus metrics.
- Auto-retrain trigger (scheduler). A GitHub Actions cron runs in the cloud and cannot reach our local MLflow/Postgres, so the trigger should run locally (e.g. a scheduler service in docker-compose).

## 8. Next tasks
- Ceylin: open the PR for branch "feature/auto-retrain" and get CI green.
- Bahadır: Prometheus (/metrics endpoint, prometheus service in docker-compose).
- Together: build the auto-retrain trigger as a local scheduler, run an end-to-end test (crawler -> /api/predict -> drift -> retrain -> reload), then README, report and presentation.
- Together (small PR): update CLAUDE.md with the tests/ rule and the requirements-api.txt note (see section 4).
