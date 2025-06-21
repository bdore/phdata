import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pickle
from contextlib import asynccontextmanager
from pandas import read_csv

# List all model_*.pkl files in the model directory.
MODEL_DIR = "model"  # Directory where model files are stored
DATA_DIR = "data"  # Directory where data files are stored
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


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
        app.state.demographics_data = demographics_data
        logger.info("### Demographics data loaded successfully.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Demographics data not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Load the model when the application starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    models_versions_list = [
        f.split("_")[1].split(".")[0]
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".pkl")
    ]

    if not models_versions_list:
        raise HTTPException(
            status_code=500, detail="No model files found in the model directory"
        )

    latest_model = max(
        models_versions_list
    )  # latest version based on naming convention
    logger.info(f"### Latest model version found: {latest_model}")
    latest_model_file = f"model_{latest_model}.pkl"
    app.state.model = None  # Initialize model state
    app.state.latest_model = latest_model_file

    try:
        app.state.model = load_model(latest_model_file)
        logger.info(f"### Model {latest_model_file} loaded successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        load_demographics_data()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load demographics data: {str(e)}"
        )

    yield


app = FastAPI(lifespan=lifespan)


# Define the data model for the request body
class RequestData(BaseModel):
    name: str
    value: int
    description: Optional[str] = None


@app.post("/predict")
async def create_data(data: RequestData):
    try:
        # Process the received data here
        return {
            "status": "success",
            "message": f"Received data for {data.name}",
            "data": data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)
