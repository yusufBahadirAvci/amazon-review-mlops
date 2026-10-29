# syntax=docker/dockerfile:1

# ---- Builder: install the API packages into a virtual environment ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
# requirements.txt is only used as constraints, so all versions come from it
COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements-api.txt -c requirements.txt


# ---- Runtime: only the virtual environment and the app code ----
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
WORKDIR /srv
# Owned by root and only readable by appuser, so the running app cannot change its code
COPY app/ ./app/

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
