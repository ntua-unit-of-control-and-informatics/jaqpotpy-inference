from pydantic import BaseModel
from typing import Any

class PredictionRequestPydantic(BaseModel):
    dataset: Any
    model: Any
    doa: Any = None
    extraConfig: Any = None