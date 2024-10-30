# This is for my local path
# from pathlib import Path
# import sys

# path_root = Path(__file__).parents[1]
# print(path_root)
# sys.path.append(str(path_root) + "/jaqpotpy")

import uvicorn
from fastapi import FastAPI

from src.api.openapi.models.prediction_request import PredictionRequest
from src.handlers.predict_sklearn import sklearn_post_handler
from src.handlers.predict_pyg import graph_post_handler
from fastapi.responses import JSONResponse
from src.loggers.log_middleware import LogMiddleware

app = FastAPI()
app.add_middleware(LogMiddleware)


@app.get("/")
def health_check():
    return {"status": "UP"}


@app.post("/predict/")
def predict(req: PredictionRequest):
    if req.model.type == "SKLEARN":
        return JSONResponse(content=sklearn_post_handler(req))
    else:
        return JSONResponse(content=graph_post_handler(req))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_config=None)
