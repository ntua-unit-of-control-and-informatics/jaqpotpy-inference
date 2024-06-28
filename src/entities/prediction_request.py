from pydantic import BaseModel
from typing import Any


class PredictionRequest:

    def __init__(self, dataset, model, doaMatrix=None):
        self.dataset = dataset
        self.model = model
        self.rawModel = model.rawModel
        self.additionalInfo = model.additionalInfo
        self.doaMatrix = doaMatrix


class PredictionRequestPydantic(BaseModel):
    dataset: Any
    model: Any
    doaMatrix: Any = None
