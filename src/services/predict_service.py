import boto3
from base64 import b64decode

from jaqpotpy.inference.service import PredictionService
from jaqpotpy.offline.offline_model_data import OfflineModelData
from jaqpot_api_client import ModelType, PredictionRequest, PredictionResponse
from src.loggers.logger import logger
from src.config.config import Settings


def _download_model_from_s3(model_id: str, model_type: ModelType) -> bytes:
    """Download model from S3 when raw model is null"""
    settings = Settings()
    s3_client = boto3.client("s3")

    try:
        response = s3_client.get_object(
            Bucket=settings.models_s3_bucket_name, Key=str(model_id)
        )
        return response["Body"].read()
    except Exception as e:
        logger.error(f"Failed to download model {model_id} from S3: {e}")
        raise


def _convert_request_to_model_data(req: PredictionRequest) -> OfflineModelData:
    """Convert PredictionRequest to OfflineModelData"""
    # Get model bytes - either from request or download from S3
    if req.model.raw_model:
        model_bytes = b64decode(req.model.raw_model)
    else:
        logger.info(f"Raw model is null for model {req.model.id}, downloading from S3")
        model_bytes = _download_model_from_s3(req.model.id, req.model.type)

    # Get preprocessor_bytes if available
    preprocessor_bytes = None
    if req.model.raw_preprocessor:
        preprocessor_bytes = b64decode(req.model.raw_preprocessor)

    # Create OfflineModelData
    return OfflineModelData(
        model_id=req.model.id,
        model_bytes=model_bytes,
        preprocessor=preprocessor_bytes,
        model_metadata=req.model,
        doas=req.model.doas,
    )


def run_prediction(req: PredictionRequest) -> PredictionResponse:
    logger.info(f"Prediction request for model {req.model.id}")

    # Convert request to model data
    model_data = _convert_request_to_model_data(req)

    # Use jaqpotpy unified prediction service
    prediction_service = PredictionService()
    response = prediction_service.predict(model_data, req.dataset, req.model.type.value)

    return response
