from typing import List, Optional
from pydantic import BaseModel

class EntryId(BaseModel):

    name : Optional[str]
    ownerUUID : Optional[str]
    URI : Optional[str]
    type : Optional[str]
