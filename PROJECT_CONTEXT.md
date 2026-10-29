# PROJECT CONTEXT — read fully before helping

Two students on a university MLOps project: Ceylin (data and model) and Bahadır (application and infrastructure). We are beginners: little Java knowledge, new to Python. Explain simply, work in small steps, and warn early if something will cause problems later.

## 1. Goal
End-to-end MLOps system that classifies Amazon product reviews by sentiment (negative / neutral / positive).
- Training data: Hugging Face "McAuley-Lab/Amazon-Reviews-2023" (English, 1-5 star ratings).
- Live data: web crawler collecting English reviews from English Amazon sites (amazon.com, amazon.co.uk, amazon.ca, amazon.com.au, amazon.in). Used for drift monitoring, not for initial training.
No tool or language restrictions from the professor; the result is what matters.

## 2. Decisions already made
- Language: Python 3.12.10 (Java version kept on branch "java-backup", do not use it).
- IDE: PyCharm, plus Claude Code inside the IDE.
- Database: PostgreSQL 15 in Docker. MLflow server v2.11.1 in Docker (port 5000). Both in docker-compose.yml.
- Secrets: .env (never committed); .env.example shows the keys. Default DB port 5432 (Bahadır uses 5433 in his own .env).
- Code quality: pre-commit with ruff (lint + format) and pytest.
- Web API: FastAPI.
- Crawler: requests + BeautifulSoup, live mode + fallback mode parsing a saved HTML page (we do NOT bypass Amazon's bot blocking).
- pandas is pinned to 2.2.3 because mlflow 2.11.1 does not support pandas 3.

## 3. Architecture (5 stages)
1. SETUP: GitHub repo, Docker Compose, pre-commit, Python skeleton.
2. DATA: DVC for dataset versioning, Great Expectations for validation (text not null, rating between 1 and 5, label in the 3 classes), crawler storing reviews in PostgreSQL.
3. MODEL: sentiment model with Optuna tuning, logged to MLflow Tracking and Model Registry.
4. DEPLOY: FastAPI /predict loading the model from MLflow, Docker multi-stage build, CI/CD with GitHub Actions.
   Note: no Docker Hub account yet; use GitHub Container Registry (ghcr.io) with the GitHub account if images must be pushed.
5. MONITOR: Prometheus for CPU/RAM/request metrics, Evidently for data drift (crawled reviews vs. training data), auto-retrain on drift or performance drop.

## 4. Roles
Bahadır (folder app/): FastAPI app, crawler, database tables, /predict endpoint, Dockerfile, GitHub Actions CI/CD, Prometheus.
Ceylin (folders data/ and training/): training dataset + DVC (raw data must NOT go into Git), Great Expectations suite, training script with Optuna + MLflow (register the best model), Evidently drift report.
Together: auto-retrain loop (Bahadır's system triggers Ceylin's training script).

## 5. Contract between our parts (do not change without telling the other person)
- Labels from stars: 1-2 = negative, 3 = neutral, 4-5 = positive.
- MLflow registered model name: "sentiment-model"; the model ready for use gets the alias "production".
- The API loads it with: mlflow.pyfunc.load_model("models:/sentiment-model@production"). Input: pandas DataFrame with a "text" column of raw review text. Output: the label string. Preprocessing lives inside the model (e.g. scikit-learn Pipeline with TF-IDF + classifier).
- MLflow tracking URI: http://localhost:5000, read from the environment variable MLFLOW_TRACKING_URI.
- Crawled reviews: PostgreSQL table "reviews" with columns including text, rating, source, marketplace, crawled_at. The drift analysis reads from this table.
- Start with a simple, fast model (TF-IDF + LogisticRegression or LinearSVC, tuned by Optuna).
- Client version pinned to match the server: mlflow==2.11.1.

## 6. Git workflow
- Never push directly to main. Work on own branches, merge through Pull Requests.
- Shared files (requirements.txt, docker-compose.yml, .gitignore, .pre-commit-config.yaml, PROJECT_CONTEXT.md): tell the other person before changing them.

## 7. Current status (last updated: 2026-10-29)
- Done (Bahadır, on main): Python skeleton (FastAPI /health, routes, DB models), crawler (parser, rating, service, tests with saved Amazon / captcha pages).
- Done (Ceylin, branch "data-dvc", PR #2 open): data/download_data.py builds data/raw/amazon_reviews.csv — 30,000 reviews from 10 product categories (All_Beauty, Amazon_Fashion, Appliances, Baby_Products, Cell_Phones_and_Accessories, Electronics, Home_and_Kitchen, Office_Products, Sports_and_Outdoors, Toys_and_Games), 1,000 per class per category, sampled from the first 100k reviews of each category, RANDOM_SEED=42. Columns: text, rating, label, category, source. Tracked with DVC (no DVC remote yet; regenerate with: python data/download_data.py).
- Dropped: Cleanlab/amazon-reviews (no star ratings, no neutral class) and a Turkish dataset.
- Done (Ceylin, branch "training"): data/validate_data.py (Great Expectations, 10 checks) and training/train.py (TF-IDF + LogisticRegression/LinearSVC, 20 Optuna trials, experiment "sentiment-training"). "sentiment-model" version 2 is "production": test macro F1 0.70 (neutral 0.59 is the weakest class). Contract test with mlflow.pyfunc.load_model("models:/sentiment-model@production") passes.
- Changed shared files: docker-compose.yml uses --artifacts-destination (models are uploaded to the server); requirements.txt pins pytest 8.3.5, virtualenv 20.29.3 and setuptools 80.9.0 because mlflow 2.11.1 needs packaging<24 and pkg_resources.
- Done (Ceylin, branch "monitoring"): monitoring/drift_report.py (Evidently 0.7.23). Compares training data with current reviews on text_length, word_count, rating and model prediction. --source db reads the "reviews" table (needs at least 50 rows); --source sample uses Amazon-Reviews-2023 categories not used in training (Video_Games, Pet_Supplies). Writes HTML + JSON (drift_detected, drifted_columns_share) to monitoring/reports/ (gitignored) and logs to the MLflow experiment "drift-monitoring". Sample run: all 4 columns drifted (share 1.0), because training data is balanced but real reviews are about 66% positive.
- requirements.txt: evidently 0.7.23 pulled protobuf 7, so protobuf is pinned to 4.25.9 and opentelemetry-proto to 1.27.0 (mlflow 2.11.1 needs protobuf<5). This is the 3rd conflict caused by the old mlflow 2.11.1; consider upgrading MLflow server + client together.
- Update (Ceylin, branch "monitoring"): drift_detected now uses only TRIGGER_COLUMNS (text_length, word_count). rating and prediction are reported as info only, because the balanced training data always differs from real reviews. Per-column scores are in the JSON ("column_drift") and in MLflow (drift_score_<column>). p-value tests (small samples) are handled: drift if p < threshold. Added 19 tests (tests/test_data_pipeline.py, test_training.py, test_drift.py; new files only); 64 tests pass in total.
- DVC remote (Ceylin): DagsHub repo "ceylinkucuk/amazon-review-mlops-data", remote name "dagshub" (default), URL https://dagshub.com/ceylinkucuk/amazon-review-mlops-data.dvc. Data is pushed (dvc push). Credentials are set per person with "dvc remote modify dagshub --local ..." (stored in .dvc/config.local, never committed). To get the data: be added as a collaborator on the DagsHub repo, set your own token locally, then "dvc pull".

- Done (Bahadır, 29 Oct): no Turkish in the project. The crawler accepts only English marketplaces (amazon.com, amazon.co.uk, amazon.ca, amazon.com.au, amazon.in; others get 400), parses ratings from the star CSS class or the English "x out of 5 stars" label only, and skips translated reviews and "reviews from other countries" with a warning (new counter skipped_foreign). Turkish test fixture removed.
- Done (Bahadır, 29 Oct): POST /api/predict ({"texts": [...]}, 1-100 texts, max 5000 chars each -> {"predictions": [...], "model_version": "..."}) loads models:/sentiment-model@production once at startup (MLFLOW_TRACKING_URI, MODEL_URI settings); POST /api/model/reload reloads it for auto-retrain (a failed reload keeps the old model); /health shows "model" and "model_version". If MLflow is down the app still starts and /predict returns 503.
- Done (Bahadır, 29 Oct): Docker + CI. Multi-stage Dockerfile (python:3.12-slim, non-root, uvicorn on 8000) installs requirements-api.txt (API runtime packages, no versions) with "-c requirements.txt", so all versions still come from requirements.txt. New "api" service in docker-compose.yml (talks to postgres-db:5432 and http://mlflow-server:5000 inside Docker, waits up to 60 s for MLflow at startup, healthcheck on /health): "docker compose up -d --build" -> http://localhost:8000. GitHub Actions (.github/workflows/ci.yml) on PRs and pushes to main: ruff, ruff format --check, pytest, docker build; on main it also pushes ghcr.io/yusufbahadiravci/amazon-review-mlops.

- Not started: Dockerfile, CI/CD, Prometheus, /predict, auto-retrain.

## 8. Next tasks
Ceylin: Great Expectations checks (branch "data-dvc"), then training script with Optuna + MLflow (branch "training").
Bahadır: review and merge PR #2; /predict endpoint once "sentiment-model@production" exists.
