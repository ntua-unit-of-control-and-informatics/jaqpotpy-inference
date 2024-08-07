import uvicorn
from fastapi import FastAPI
from src.handlers.predict import model_post_handler
from src.entities.prediction_request import PredictionRequestPydantic
from fastapi.responses import JSONResponse
from src.loggers.log_middleware import LogMiddleware


app = FastAPI()
app.add_middleware(LogMiddleware)


@app.get('/')
def health_check():
    return {'status': 'UP'}


@app.post('/predict/')
def predict(req: PredictionRequestPydantic):
    return JSONResponse(content=model_post_handler(req))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_config=None)
