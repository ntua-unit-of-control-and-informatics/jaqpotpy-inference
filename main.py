# This is for my local path
# from pathlib import Path
# import sys

# path_root = Path(__file__).parents[1]
# print(path_root)
# sys.path.append(str(path_root) + "/jaqpotpy")

import uvicorn
from fastapi import FastAPI
from jaqpotpy.api.openapi import PredictionRequest, PredictionResponse, ModelType

from src.handlers.predict_sklearn_onnx import sklearn_onnx_post_handler
from src.handlers.predict_torch import torch_post_handler

from src.loggers.logger import logger
from src.loggers.log_middleware import LogMiddleware

app = FastAPI()
app.add_middleware(LogMiddleware)


@app.get("/")
def health_check():
    return {"status": "UP"}


@app.post("/predict")
def predict(req: PredictionRequest) -> PredictionResponse:
    logger.info("Prediction request for model " + str(req.model.id))

    match req.model.type:
        case ModelType.SKLEARN_ONNX:
            return sklearn_onnx_post_handler(req)
        case (
            ModelType.TORCH_GEOMETRIC_ONNX,
            ModelType.TORCHSCRIPT,
            ModelType.TORCH_SEQUENCE_ONNX,
        ):
            return torch_post_handler(req)
        case _:
            raise Exception("Model type not supported")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_config=None)
