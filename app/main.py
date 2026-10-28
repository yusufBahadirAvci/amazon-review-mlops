from typing import Annotated

from fastapi import Depends, FastAPI, Response, status

from app.db import database_is_up

app = FastAPI(title="Product Review Classifier")


@app.get("/health")
def health(response: Response, db_up: Annotated[bool, Depends(database_is_up)]) -> dict[str, str]:
    if not db_up:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if db_up else "degraded", "database": "up" if db_up else "down"}
