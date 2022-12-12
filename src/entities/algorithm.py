from typing import List, Optional
from pydantic import BaseModel
from .meta import MetaInfo
from .parameter import Parameter

class Algorithm(BaseModel):
    meta : Optional[MetaInfo]
    ontological_classes : Optional[List[str]]
    visible : Optional[bool]
    temporary : Optional[str]
    featured : Optional[str]
    parameters : Optional[Parameter]
    ranking : Optional[str]
    bibtex : Optional[dict]
    training_service : Optional[str]
    prediction_service : Optional[str]
    report_service : Optional[str]
    _id : str
