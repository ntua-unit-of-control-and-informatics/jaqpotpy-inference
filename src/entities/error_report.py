from .meta import MetaInfo  # noqa: F401,E501
from typing import List, Optional
from pydantic import BaseModel


class ErrorReport(BaseModel):

    meta : Optional[MetaInfo]
    ontological_classes : Optional[List[str]]
    visible : Optional[bool]
    temporary : Optional[str]
    featured : Optional[str]
    code : Optional[str]
    actor : Optional[str]
    message : Optional[str]
    details : Optional[List[str]]
    http_status : Optional[int]
    id : str
    discriminator : Optional[str]
