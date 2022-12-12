from .error_report import ErrorReport  # noqa: F401,E501
from .meta import MetaInfo  # noqa: F401,E501
from typing import List, Optional
from pydantic import BaseModel

class Task(BaseModel):

    meta : Optional[MetaInfo]
    ontological_classes : Optional[List[str]]
    visible : Optional[bool]
    temporary : Optional[str]
    featured : Optional[str]
    result_uri : Optional[str]
    result : Optional[str]
    percentage_completed : Optional[float]
    error_report : Optional[ErrorReport]
    http_status : Optional[int]
    duration : Optional[float]
    type : Optional[str]
    id : str
    status : Optional[str]
    discriminator : Optional[str]
