import base64

import onnx
from jaqpot_api_client import PredictionRequest

from src.config.config import settings
from src.helpers.s3_client import S3Client

models_s3_client = S3Client(bucket_name=settings.models_s3_bucket_name)


def retrieve_onnx_model_from_request(request: PredictionRequest):
    if request.model.raw_model is None:
        file_obj, error = models_s3_client.download_file(str(request.model.id))
        if file_obj is None:
            raise Exception(f"Failed to download model: {error}")
        model = onnx.load(file_obj)
    else:
        raw_model = base64.b64decode(request.model.raw_model)
        model = onnx.load_from_string(raw_model)
    return model
