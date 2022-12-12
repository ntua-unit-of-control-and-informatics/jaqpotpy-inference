from typing import List, Optional
from pydantic import BaseModel

class PretrainedModel(BaseModel):

    raw_model : Optional[str]
    pmml_model : Optional[str]
    additional_info : Optional[List[str]]
    dependent_features : Optional[List[str]]
    independent_features : Optional[List[str]]
    predicted_features : Optional[List[str]]
    implemented_in : Optional[str]
    implemented_with : Optional[str]
    title : Optional[str]
    discription : Optional[str]
    algorithm : Optional[str]
    discriminator : Optional[str]
