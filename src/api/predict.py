from fastapi import APIRouter
from jaqpot_api_client import PredictionResponse, PredictionRequest

from src.services.predict_service import run_prediction

router = APIRouter()


@router.post("", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    return run_prediction(req)
