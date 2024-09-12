from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
print(path_root)
sys.path.append(str(path_root) + "/jaqpotpy")

import uvicorn
from fastapi import FastAPI
from src.handlers.predict import model_post_handler, graph_post_handler
from src.entities.prediction_request import PredictionRequestPydantic
from fastapi.responses import JSONResponse
from src.loggers.log_middleware import LogMiddleware

app = FastAPI()
app.add_middleware(LogMiddleware)


@app.get("/")
def health_check():
    return {"status": "UP"}


@app.post("/predict/")
def predict(req: PredictionRequestPydantic):
    if req.model["type"] == "SKLEARN":
        return JSONResponse(content=model_post_handler(req))
    elif req.model["type"] == "TORCH":
        return JSONResponse(content=graph_post_handler(req))
    else:
        raise ValueError("Only SKLEARN and TORCH models are supported")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_config=None)
