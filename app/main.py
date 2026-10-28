import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.crawler import CrawlerBlockedError, CrawlerError, InvalidCrawlInputError
from app.db import Base, database_is_up, engine
from app.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        Base.metadata.create_all(engine)
    except SQLAlchemyError:
        # Keep serving so /health can report the database as down
        logger.exception("Could not create database tables; is PostgreSQL running?")
    yield


app = FastAPI(title="Product Review Classifier", lifespan=lifespan)
app.include_router(router)


def _error(status_code: int, message: str, hint: str | None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message, "hint": hint})


@app.exception_handler(CrawlerBlockedError)
def handle_blocked(_: Request, exc: CrawlerBlockedError) -> JSONResponse:
    return _error(status.HTTP_503_SERVICE_UNAVAILABLE, exc.message, exc.hint)


@app.exception_handler(InvalidCrawlInputError)
def handle_invalid_input(_: Request, exc: InvalidCrawlInputError) -> JSONResponse:
    return _error(status.HTTP_400_BAD_REQUEST, exc.message, exc.hint)


@app.exception_handler(CrawlerError)
def handle_crawler_error(_: Request, exc: CrawlerError) -> JSONResponse:
    return _error(status.HTTP_502_BAD_GATEWAY, exc.message, exc.hint)


@app.exception_handler(RequestValidationError)
def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    message = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'][1:])}: {error['msg']}"
        for error in exc.errors()
    )
    return _error(
        status.HTTP_400_BAD_REQUEST, message, "Check the request body and query parameters."
    )


@app.get("/health")
def health(response: Response, db_up: Annotated[bool, Depends(database_is_up)]) -> dict[str, str]:
    if not db_up:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if db_up else "degraded", "database": "up" if db_up else "down"}
