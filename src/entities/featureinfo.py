from typing import List, Optional
from pydantic import BaseModel

class FeatureInfo(BaseModel):

    name : Optional[str]
    units : Optional[str]
    conditions : Optional[List[str]]
    category : Optional[str]
    ont : Optional[str]
    uri : Optional[str]
    key : Optional[str]
    