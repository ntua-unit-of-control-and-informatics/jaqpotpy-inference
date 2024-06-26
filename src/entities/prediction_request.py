from pydantic import BaseModel
from typing import Any


class PredictionRequest:

    def __init__(self, dataset, model, rawModel, additionalInfo=None, doaMatrix=None):
        self.dataset = dataset
        self.model = model
        self.rawModel = rawModel
        self.additionalInfo = additionalInfo
        self.doaMatrix = doaMatrix


class PredictionRequestPydantic(BaseModel):
    dataset: Any
    model: Any
    additionalInfo: Any
    doaMatrix: Any = None
    rawModel: Any
