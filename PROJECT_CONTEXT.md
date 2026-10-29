# PROJECT CONTEXT — read fully before helping

Two students on a university MLOps project: Ceylin (data and model) and Bahadır (application and infrastructure). We are beginners: little Java knowledge, new to Python. Explain simply, work in small steps, and warn early if something will cause problems later.

## 1. Goal
End-to-end MLOps system that classifies Amazon product reviews by sentiment (negative / neutral / positive).
- Training data: Hugging Face "McAuley-Lab/Amazon-Reviews-2023" (English, 1-5 star ratings).
- Live data: web crawler collecting reviews from Amazon country sites (including amazon.com.tr). Used for drift monitoring, not for initial training.
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
- Done (Bahadır, on main): Python skeleton (FastAPI /health, routes, DB models), crawler (parser, rating, service, tests with saved Amazon / amazon.com.tr / captcha pages).
- Done (Ceylin, branch "data-dvc", PR #2 open): data/download_data.py builds data/raw/amazon_reviews.csv — 30,000 reviews from 10 product categories (All_Beauty, Amazon_Fashion, Appliances, Baby_Products, Cell_Phones_and_Accessories, Electronics, Home_and_Kitchen, Office_Products, Sports_and_Outdoors, Toys_and_Games), 1,000 per class per category, sampled from the first 100k reviews of each category, RANDOM_SEED=42. Columns: text, rating, label, category, source. Tracked with DVC (no DVC remote yet; regenerate with: python data/download_data.py).
- Dropped: Cleanlab/amazon-reviews (no star ratings, no neutral class) and a Turkish dataset.
- Done (Ceylin, branch "training"): data/validate_data.py (Great Expectations, 10 checks) and training/train.py (TF-IDF + LogisticRegression/LinearSVC, 20 Optuna trials, experiment "sentiment-training"). "sentiment-model" version 2 is "production": test macro F1 0.70 (neutral 0.59 is the weakest class). Contract test with mlflow.pyfunc.load_model("models:/sentiment-model@production") passes.
- Changed shared files: docker-compose.yml uses --artifacts-destination (models are uploaded to the server); requirements.txt pins pytest 8.3.5, virtualenv 20.29.3 and setuptools 80.9.0 because mlflow 2.11.1 needs packaging<24 and pkg_resources.
- Not started: Dockerfile, CI/CD, Prometheus, /predict, Evidently, auto-retrain.

## 8. Next tasks
Ceylin: Great Expectations checks (branch "data-dvc"), then training script with Optuna + MLflow (branch "training").
Bahadır: review and merge PR #2; /predict endpoint once "sentiment-model@production" exists.
