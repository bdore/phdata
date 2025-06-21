import logging
import os
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import pickle
from contextlib import asynccontextmanager
from pandas import read_csv, DataFrame


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


class Prediction(BaseModel):
    price: float


class PredictionSubset(BaseModel):
    bedrooms: int
    bathrooms: int
    sqft_living: float
    sqft_lot: float
    floors: int
    sqft_above: float
    sqft_basement: float
    zipcode: str


def load_model(model_name: str):
    """Load a model from a pickle file."""
    try:
        with open(f"{MODEL_DIR}/{model_name}", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_demographics_data():
    """Load demographics data from a CSV file."""
    demographics_path = os.path.join(DATA_DIR, "zipcode_demographics.csv")
    try:
        demographics_data = read_csv(demographics_path, dtype={"zipcode": str})
        return demographics_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Demographics data not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_features():
    """Load features from a JSON file."""
    features_path = os.path.join(
        MODEL_DIR, f"model_{app.state.latest_model}_features.json"
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

    app.state.latest_model = max(models_versions_list)
    logger.info(f"### Latest model version found: {app.state.latest_model}")
    app.state.latest_model_filename = f"model_{app.state.latest_model}.pkl"
    app.state.model = None

    try:
        app.state.model = load_model(app.state.latest_model_filename)
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


@app.post("/predict")
async def make_prediction(request_data: Request) -> List[Prediction]:
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
    return [Prediction(price=price) for price in predictions]

    # try:
    #     input_data = {
    #         "name": [request_data.name],
    #         "value": [request_data.value],
    #         "description": [request_data.description or ""],
    #     }
    #     input_df = read_csv(input_data)
    #     prediction = app.state.model.predict(input_df)
    #     return {"prediction": prediction.tolist()}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))


# @app.post("/predict_subset")
# async def make_prediction_subset(request_data: Request) -> List[Prediction]:
