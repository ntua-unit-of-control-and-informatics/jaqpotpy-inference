from typing import List, Optional
from pydantic import BaseModel
import datetime

class MetaInfo(BaseModel):
    identifiers : Optional[List[str]]
    comments : Optional[List[str]]
    descriptions : Optional[List[str]]
    titles : Optional[List[str]]
    subjects : Optional[List[str]]
    publishers : Optional[List[str]]
    creators : Optional[List[str]]
    contributors : Optional[List[str]]
    audiences : Optional[List[str]]
    rights : Optional[List[str]]
    sameAs : Optional[str]
    seeAlso : Optional[str]
    hasSources : Optional[List[str]]
    doi : Optional[str]
    date : Optional[datetime.datetime]
