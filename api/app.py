import logging
import os
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from typing import List
import pickle
from contextlib import asynccontextmanager
from pandas import read_csv, DataFrame
from sklearn.pipeline import Pipeline


MODEL_DIR = "model"
DATA_DIR = "data"
SALES_COLUMN_SELECTION = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


class UnseenHousingDataSubset(BaseModel):
    bedrooms: int
    bathrooms: int
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: str

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "bedrooms": 4,
                "bathrooms": 1,
                "sqft_living": 1680,
                "sqft_lot": 5043,
                "floors": 1.5,
                "sqft_above": 1680,
                "sqft_basement": 0,
                "zipcode": "98118",
            }
        },
    )


class UnseenHousingData(UnseenHousingDataSubset):
    waterfront: int
    view: int
    condition: int
    grade: int
    yr_built: int
    yr_renovated: int
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bedrooms": 4,
                "bathrooms": 1.0,
                "sqft_living": 1680,
                "sqft_lot": 5043,
                "floors": 1.5,
                "waterfront": 0,
                "view": 0,
                "condition": 4,
                "grade": 6,
                "sqft_above": 1680,
                "sqft_basement": 0,
                "yr_built": 1911,
                "yr_renovated": 0,
                "zipcode": "98118",
                "lat": 47.5354,
                "long": -122.273,
                "sqft_living15": 1560,
                "sqft_lot15": 5765,
            }
        },
    )


class Model(BaseModel):
    version: int

    model_config = ConfigDict(
        json_schema_extra={"example": {"version": 2}},
    )


class Prediction(BaseModel):
    price: float


class PredictionResponse(BaseModel):
    predictions: List[Prediction]
    model: Model


class ApiMessage(BaseModel):
    message: str


def load_model(model_version: int) -> Pipeline:
    """Load a model from a pickle file."""
    filename = f"model_{model_version}.pkl"
    try:
        with open(f"{MODEL_DIR}/{filename}", "rb") as f:
            model = pickle.load(f)

        app.state.model = model
        app.state.model_version = model_version
        app.state.model_filename = filename

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_demographics_data() -> DataFrame:
    """Load demographics data from a CSV file."""
    demographics_path = os.path.join(DATA_DIR, "zipcode_demographics.csv")
    try:
        demographics_data = read_csv(demographics_path, dtype={"zipcode": str})
        return demographics_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Demographics data not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_features() -> dict:
    """Load features from a JSON file."""
    features_path = os.path.join(
        MODEL_DIR, f"model_{app.state.model_version}_features.json"
    )
    try:
        with open(features_path, "r") as f:
            features = json.load(f)
        return features
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Features file not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load features: {str(e)}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_versions_list = [
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".pkl")
    ]

    if not models_versions_list:
        raise HTTPException(
            status_code=500, detail="No model files found in the model directory"
        )

    app.state.models_qty = len(models_versions_list)
    logger.info(f"### Number of model versions found: {app.state.models_qty}")
    app.state.latest_model_filename = f"model_{app.state.models_qty}.pkl"
    app.state.model = None

    try:
        load_model(model_version=1)
        logger.info(f"### Model {app.state.model_filename} loaded successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        app.state.demographics_data = load_demographics_data()
        logger.info("### Demographics data loaded successfully.")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load demographics data: {str(e)}"
        )

    try:
        app.state.features = load_features()
        logger.info("### Features loaded successfully.")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load features: {str(e)}"
        )

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predictions/all")
async def make_prediction(
    unseen_data_list: List[UnseenHousingData],
) -> PredictionResponse:
    """Endpoint to make predictions using the loaded model."""
    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data = DataFrame([dict(x) for x in unseen_data_list])
    data_df = DataFrame(data)
    data_df = data_df[SALES_COLUMN_SELECTION]
    data_df = data_df.merge(app.state.demographics_data, how="left", on="zipcode").drop(
        columns="zipcode"
    )
    predictions = app.state.model.predict(data_df)
    return PredictionResponse(
        predictions=[Prediction(price=price) for price in predictions],
        model=Model(version=app.state.model_version),
    )


@app.post("/predictions/subset")
async def make_prediction_subset(
    prediction_subset_list: List[UnseenHousingDataSubset],
) -> PredictionResponse:
    """Endpoint to make predictions using a subset of features."""
    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data_df = DataFrame([dict(x) for x in prediction_subset_list])
    data_df = data_df.merge(app.state.demographics_data, how="left", on="zipcode").drop(
        columns="zipcode"
    )
    predictions = app.state.model.predict(data_df)
    return PredictionResponse(
        predictions=[Prediction(price=price) for price in predictions],
        model=Model(version=app.state.model_version),
    )


@app.post("/models/select")
async def select_model(model: Model) -> ApiMessage:
    """Endpoint to select a new model version."""
    try:
        load_model(model.version)
        message = f"Model version {model.version} selected successfully."
        logger.info(f"### Model version {model.version} selected successfully.")
        return ApiMessage(message=message)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to select model: {str(e)}")


@app.get("/models/list")
async def list_models() -> List[Model]:
    """Endpoint to list available model versions."""
    models_versions_list = [
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".pkl")
    ]
    models_versions_list.sort(reverse=True)

    if not models_versions_list:
        raise HTTPException(status_code=404, detail="No models found")

    return [Model(version=version) for version in models_versions_list]
