# coding: utf-8

"""
    Jaqpot API

    A modern RESTful API for model management and prediction services, built using Spring Boot and Kotlin. Supports seamless integration with machine learning workflows.

    The version of the OpenAPI document: 1.0.0
    Contact: upci.ntua@gmail.com
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""  # noqa: E501


from __future__ import annotations
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, ClassVar, Dict, List, Optional
from src.api.openapi.models.binary_classification_scores import BinaryClassificationScores
from src.api.openapi.models.multiclass_classification_scores import MulticlassClassificationScores
from src.api.openapi.models.regression_scores import RegressionScores
from typing import Optional, Set
from typing_extensions import Self

class Scores(BaseModel):
    """
    Scores
    """ # noqa: E501
    regression: Optional[RegressionScores] = None
    binary_classification: Optional[BinaryClassificationScores] = Field(default=None, alias="binaryClassification")
    multiclass_classification: Optional[MulticlassClassificationScores] = Field(default=None, alias="multiclassClassification")
    __properties: ClassVar[List[str]] = ["regression", "binaryClassification", "multiclassClassification"]

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        protected_namespaces=(),
    )


    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.model_dump(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        # TODO: pydantic v2: use .model_dump_json(by_alias=True, exclude_unset=True) instead
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> Optional[Self]:
        """Create an instance of Scores from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> Dict[str, Any]:
        """Return the dictionary representation of the model using alias.

        This has the following differences from calling pydantic's
        `self.model_dump(by_alias=True)`:

        * `None` is only added to the output dict for nullable fields that
          were set at model initialization. Other fields with value `None`
          are ignored.
        """
        excluded_fields: Set[str] = set([
        ])

        _dict = self.model_dump(
            by_alias=True,
            exclude=excluded_fields,
            exclude_none=True,
        )
        # override the default output from pydantic by calling `to_dict()` of regression
        if self.regression:
            _dict['regression'] = self.regression.to_dict()
        # override the default output from pydantic by calling `to_dict()` of binary_classification
        if self.binary_classification:
            _dict['binaryClassification'] = self.binary_classification.to_dict()
        # override the default output from pydantic by calling `to_dict()` of multiclass_classification
        if self.multiclass_classification:
            _dict['multiclassClassification'] = self.multiclass_classification.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of Scores from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "regression": RegressionScores.from_dict(obj["regression"]) if obj.get("regression") is not None else None,
            "binaryClassification": BinaryClassificationScores.from_dict(obj["binaryClassification"]) if obj.get("binaryClassification") is not None else None,
            "multiclassClassification": MulticlassClassificationScores.from_dict(obj["multiclassClassification"]) if obj.get("multiclassClassification") is not None else None
        })
        return _obj


