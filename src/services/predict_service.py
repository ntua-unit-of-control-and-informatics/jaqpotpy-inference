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

    # Determine the file extension based on model type
    if model_type == ModelType.TORCHSCRIPT:
        file_extension = ".pt"
    else:
        file_extension = ".onnx"

    key = f"{model_id}/model{file_extension}"

    try:
        response = s3_client.get_object(Bucket=settings.models_s3_bucket_name, Key=key)
        return response["Body"].read()
    except Exception as e:
        logger.error(f"Failed to download model {model_id} from S3: {e}")
        raise


def _convert_request_to_model_data(req: PredictionRequest) -> OfflineModelData:
    """Convert PredictionRequest to OfflineModelData"""
    # Get model bytes - either from request or download from S3
    if req.model.raw_model:
        onnx_bytes = b64decode(req.model.raw_model)
    else:
        logger.info(f"Raw model is null for model {req.model.id}, downloading from S3")
        onnx_bytes = _download_model_from_s3(req.model.id, req.model.type)

    # Get preprocessor if available
    preprocessor = None
    if req.model.raw_preprocessor:
        import onnx

        preprocessor = onnx.load_from_string(b64decode(req.model.raw_preprocessor))

    # Create OfflineModelData
    return OfflineModelData(
        model_id=req.model.id,
        onnx_bytes=onnx_bytes,
        preprocessor=preprocessor,
        model_metadata=req.model,
    )


def run_prediction(req: PredictionRequest) -> PredictionResponse:
    logger.info(f"Prediction request for model {req.model.id}")

    # Convert request to model data
    model_data = _convert_request_to_model_data(req)

    # Use jaqpotpy unified prediction service
    prediction_service = PredictionService()
    response = prediction_service.predict(model_data, req.dataset, req.model.type.value)

    return response
