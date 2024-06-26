from pydantic import BaseModel
from typing import Any


class PredictionRequest:

    def __init__(self, dataset, model, additionalInfo=None, doaMatrix=None):
        self.dataset = dataset
        self.rawModel = model.actualModel
        self.additionalInfo = additionalInfo
        self.doaMatrix = doaMatrix


class PredictionRequestPydantic(BaseModel):
    dataset: Any
    model: Any
    additionalInfo: Any
    doaMatrix: Any = None
