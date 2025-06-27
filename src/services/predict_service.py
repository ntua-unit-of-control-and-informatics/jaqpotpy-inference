from src.loggers.logger import logger
from src.config.config import settings
from src.helpers.s3_client import S3Client
from jaqpot_api_client import ModelType, PredictionRequest, PredictionResponse
from jaqpotpy.inference.service import PredictionService

# Initialize S3 client for model downloads
models_s3_client = S3Client(bucket_name=settings.models_s3_bucket_name)

# Global prediction service instance using jaqpotpy shared logic
prediction_service = PredictionService(local_mode=False, s3_client=models_s3_client)


def run_prediction(req: PredictionRequest) -> PredictionResponse:
    """
    Simplified prediction service using jaqpotpy shared logic.

    This function now delegates all prediction logic to the unified
    PredictionService in jaqpotpy, ensuring consistency between
    local and production inference.
    """
    logger.info(
        f"Prediction request for model {req.model.id} using shared jaqpotpy logic"
    )

    try:
        # Use the unified prediction service
        response = prediction_service.predict(req)
        logger.info(f"Prediction completed successfully for model {req.model.id}")
        return response
    except Exception as e:
        logger.error(f"Prediction failed for model {req.model.id}: {str(e)}")
        raise
