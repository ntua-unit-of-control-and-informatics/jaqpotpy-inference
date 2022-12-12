from typing import List, Optional
from pydantic import BaseModel

class Parameter(BaseModel):
    required : Optional[str]
    description : Optional[str]
    vendor_extensions : Optional[List[str]]
    pattern : Optional[str]
    _in : Optional[str]
    name : Optional[str]
    discriminator : Optional[str]
