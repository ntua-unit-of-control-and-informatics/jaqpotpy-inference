from .jaqpot_base import JaqpotEntity
from .dataentry import DataEntry
from .featureinfo import FeatureInfo
from .meta import MetaInfo
from typing import List, Optional
from pydantic import BaseModel

class Dataset(BaseModel):
    meta : Optional[MetaInfo]
    ontologicalClasses : Optional[List[str]]
    visible : Optional[bool]
    temporary : Optional[str]
    featured : Optional[str]
    datasetUri : Optional[str]
    byModel : Optional[str]
    dataEntry : Optional[DataEntry]
    features : Optional[FeatureInfo]
    totalRows : Optional[int]
    totalColumns : Optional[int]
    descriptors : Optional[List[str]]
    id : str