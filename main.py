import uvicorn
from fastapi import FastAPI
from src.loggers.log_middleware import LogMiddleware
from src.api.predict import router as predict_router

app = FastAPI()
app.add_middleware(LogMiddleware)

app.include_router(predict_router, prefix="/predict")


@app.get("/")
def health_check():
    return {"status": "UP"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_config=None)
