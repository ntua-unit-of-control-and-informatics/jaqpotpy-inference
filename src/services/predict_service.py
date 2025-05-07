from src.handlers.predict_sklearn_onnx import sklearn_onnx_post_handler
from src.handlers.predict_torch_onnx import torch_onnx_post_handler
from src.handlers.predict_torch_geometric import torch_geometric_post_handler
from src.handlers.predict_torch_sequence import torch_sequence_post_handler
from src.loggers.logger import logger
from jaqpot_api_client import ModelType, PredictionRequest, PredictionResponse


def run_prediction(req: PredictionRequest) -> PredictionResponse:
    logger.info(f"Prediction request for model {req.model.id}")

    match req.model.type:
        case ModelType.SKLEARN_ONNX:
            return sklearn_onnx_post_handler(req)
        case ModelType.TORCH_ONNX:
            return torch_onnx_post_handler(req)
        case ModelType.TORCH_SEQUENCE_ONNX:
            return torch_sequence_post_handler(req)
        case ModelType.TORCH_GEOMETRIC_ONNX | ModelType.TORCHSCRIPT:
            return torch_geometric_post_handler(req)
        case _:
            raise ValueError("Model type not supported")
