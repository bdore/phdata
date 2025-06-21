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


MODEL_DIR = "model"  # Directory where model files are stored
DATA_DIR = "data"  # Directory where data files are stored
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


class PredictionData(BaseModel):
    features: list = SALES_COLUMN_SELECTION


class PredictionSubset(BaseModel):
    bedrooms: int
    bathrooms: int
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: str

    model_config = ConfigDict(extra="forbid")


class Model(BaseModel):
    version: int


class Prediction(BaseModel):
    price: float


class ApiResponse(BaseModel):
    model: Model
    predictions: List[Prediction]


def load_model(model_version: int) -> Pipeline:
    """Load a model from a pickle file."""

    try:
        with open(f"{MODEL_DIR}/model_{model_version}.pkl", "rb") as f:
            model = pickle.load(f)
        return model
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
        MODEL_DIR, f"model_{app.state.latest_model_version}_features.json"
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

    app.state.latest_model_version = max(models_versions_list)
    logger.info(f"### Latest model version found: {app.state.latest_model_version}")
    app.state.latest_model_filename = f"model_{app.state.latest_model_version}.pkl"
    app.state.model = None

    try:
        app.state.model = load_model(app.state.latest_model_version)
        logger.info(f"### Model {app.state.latest_model_filename} loaded successfully.")
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
async def make_prediction(request_data: Request) -> ApiResponse:
    """Endpoint to make predictions using the loaded model."""
    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data = await request_data.json()
    data_df = DataFrame(data)
    data_df = data_df[SALES_COLUMN_SELECTION]
    data_df = data_df.merge(app.state.demographics_data, how="left", on="zipcode").drop(
        columns="zipcode"
    )
    predictions = app.state.model.predict(data_df)
    return ApiResponse(
        model=Model(version=app.state.latest_model_version),
        predictions=[Prediction(price=price) for price in predictions],
    )


@app.post("/predictions/subset")
async def make_prediction_subset(
    prediction_subset_list: List[PredictionSubset],
) -> ApiResponse:
    """Endpoint to make predictions using a subset of features."""
    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data_df = DataFrame(prediction_subset_list)
    data_df = data_df.merge(app.state.demographics_data, how="left", on="zipcode").drop(
        columns="zipcode"
    )
    predictions = app.state.model.predict(data_df)
    return ApiResponse(
        model=Model(version=app.state.latest_model_version),
        predictions=[Prediction(price=price) for price in predictions],
    )


@app.post("/models/update")
async def update_model(model: Model) -> dict:
    """Endpoint to update the model to a new version."""
    try:
        app.state.model = load_model(model.version)
        app.state.latest_model_version = model.version
        app.state.latest_model_filename = f"model_{model.version}.pkl"
        logger.info(f"### Model updated to version {model.version} successfully.")
        return {"message": f"Model updated to version {model.version} successfully."}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")


@app.get("/models/list")
async def list_models() -> List[Model]:
    """Endpoint to list available model versions."""
    models_versions_list = [
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".pkl")
    ]
    if not models_versions_list:
        raise HTTPException(status_code=404, detail="No models found")
    return [Model(version=version) for version in sorted(models_versions_list)]
