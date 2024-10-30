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

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from typing import Any, ClassVar, Dict, List, Optional
from typing_extensions import Annotated
from src.api.openapi.models.model_type import ModelType
from src.api.openapi.models.model_visibility import ModelVisibility
from src.api.openapi.models.organization_summary import OrganizationSummary
from src.api.openapi.models.user import User
from typing import Optional, Set
from typing_extensions import Self

class ModelSummary(BaseModel):
    """
    ModelSummary
    """ # noqa: E501
    id: StrictInt
    name: Annotated[str, Field(min_length=3, strict=True, max_length=255)]
    visibility: ModelVisibility
    description: Optional[Annotated[str, Field(min_length=3, strict=True, max_length=50000)]] = None
    creator: Optional[User] = None
    type: ModelType
    dependent_features_length: Optional[StrictInt] = Field(default=None, alias="dependentFeaturesLength")
    independent_features_length: Optional[StrictInt] = Field(default=None, alias="independentFeaturesLength")
    shared_with_organizations: List[OrganizationSummary] = Field(alias="sharedWithOrganizations")
    created_at: datetime = Field(description="The date and time when the feature was created.", alias="createdAt")
    updated_at: Optional[datetime] = Field(default=None, description="The date and time when the feature was last updated.", alias="updatedAt")
    __properties: ClassVar[List[str]] = ["id", "name", "visibility", "description", "creator", "type", "dependentFeaturesLength", "independentFeaturesLength", "sharedWithOrganizations", "createdAt", "updatedAt"]

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
        """Create an instance of ModelSummary from a JSON string"""
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
        # override the default output from pydantic by calling `to_dict()` of creator
        if self.creator:
            _dict['creator'] = self.creator.to_dict()
        # override the default output from pydantic by calling `to_dict()` of each item in shared_with_organizations (list)
        _items = []
        if self.shared_with_organizations:
            for _item_shared_with_organizations in self.shared_with_organizations:
                if _item_shared_with_organizations:
                    _items.append(_item_shared_with_organizations.to_dict())
            _dict['sharedWithOrganizations'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of ModelSummary from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "id": obj.get("id"),
            "name": obj.get("name"),
            "visibility": obj.get("visibility"),
            "description": obj.get("description"),
            "creator": User.from_dict(obj["creator"]) if obj.get("creator") is not None else None,
            "type": obj.get("type"),
            "dependentFeaturesLength": obj.get("dependentFeaturesLength"),
            "independentFeaturesLength": obj.get("independentFeaturesLength"),
            "sharedWithOrganizations": [OrganizationSummary.from_dict(_item) for _item in obj["sharedWithOrganizations"]] if obj.get("sharedWithOrganizations") is not None else None,
            "createdAt": obj.get("createdAt"),
            "updatedAt": obj.get("updatedAt")
        })
        return _obj


