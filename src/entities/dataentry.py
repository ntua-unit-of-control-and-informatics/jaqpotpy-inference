from .entryid import EntryId
from pydantic import BaseModel
from typing import List, Optional

class DataEntry(BaseModel):
    entryId : str
    values : Optional[List[str]]