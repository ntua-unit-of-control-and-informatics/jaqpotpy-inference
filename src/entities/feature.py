from .meta import MetaInfo  # noqa: F401,E501
from typing import List, Optional
from pydantic import BaseModel


class Feature(BaseModel):
    meta : Optional[MetaInfo]
    ontologicalClasses : Optional[List[str]]
    visible : Optional[bool]
    temporary : Optional[str]
    featured : Optional[str]
    units : Optional[str]
    predictorFor : Optional[str]
    admissibleValues : Optional[str]
    actualIndependentFeatureName : Optional[str]
    fromPretrained : Optional[str]
    id : str
    discriminator : Optional[str]
