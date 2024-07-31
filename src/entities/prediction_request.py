from pydantic import BaseModel
from typing import Any


class PredictionRequest:

    def __init__(self, dataset, model, doa=None):
        self.dataset = dataset
        self.model = model
        self.rawModel = model.rawModel
        self.additionalInfo = model.additionalInfo
        self.doa = doa


class PredictionRequestPydantic(BaseModel):
    dataset: Any
    model: Any
    doa: Any = None
