from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import AfterValidator, BaseModel, Field, StringConstraints

from app.config import ModelSettings, get_model_settings
from app.ml_model import ModelStore, get_model_store

MAX_TEXTS = 100
MAX_TEXT_LENGTH = 5000

router = APIRouter(prefix="/api")

ModelStoreDep = Annotated[ModelStore, Depends(get_model_store)]
ModelSettingsDep = Annotated[ModelSettings, Depends(get_model_settings)]


def _not_blank(text: str) -> str:
    if not text.strip():
        raise ValueError("text must not be empty or only whitespace")
    return text


ReviewText = Annotated[
    str, StringConstraints(max_length=MAX_TEXT_LENGTH), AfterValidator(_not_blank)
]


class PredictRequest(BaseModel):
    texts: Annotated[list[ReviewText], Field(min_length=1, max_length=MAX_TEXTS)]


class PredictResponse(BaseModel):
    predictions: list[str]
    model_version: str


class ReloadResponse(BaseModel):
    model_version: str
    previous_version: str | None


@router.post("/predict")
def predict(
    request: PredictRequest, store: ModelStoreDep, settings: ModelSettingsDep
) -> PredictResponse:
    loaded = store.require(settings)
    # Contract: a DataFrame with a "text" column in, one label string per row out
    labels = loaded.model.predict(pd.DataFrame({"text": request.texts}))
    return PredictResponse(
        predictions=[str(label) for label in labels], model_version=loaded.version
    )


@router.post("/model/reload")
def reload_model(store: ModelStoreDep, settings: ModelSettingsDep) -> ReloadResponse:
    previous = store.current
    loaded = store.load(settings)
    return ReloadResponse(
        model_version=loaded.version,
        previous_version=previous.version if previous is not None else None,
    )
