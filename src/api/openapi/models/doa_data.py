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
import json
import pprint
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationError,
    field_validator,
)
from typing import Any, List, Optional
from src.api.openapi.models.bounding_box_doa import BoundingBoxDoa
from src.api.openapi.models.city_block_doa import CityBlockDoa
from src.api.openapi.models.kernel_based_doa import KernelBasedDoa
from src.api.openapi.models.leverage_doa import LeverageDoa
from src.api.openapi.models.mahalanobis_doa import MahalanobisDoa
from src.api.openapi.models.mean_var_doa import MeanVarDoa
from pydantic import StrictStr, Field
from typing import Union, List, Set, Optional, Dict
from typing_extensions import Literal, Self

DOADATA_ONE_OF_SCHEMAS = [
    "BoundingBoxDoa",
    "CityBlockDoa",
    "KernelBasedDoa",
    "LeverageDoa",
    "MahalanobisDoa",
    "MeanVarDoa",
]


class DoaData(BaseModel):
    """
    The doa calculated data
    """

    # data type: LeverageDoa
    oneof_schema_1_validator: Optional[LeverageDoa] = None
    # data type: BoundingBoxDoa
    oneof_schema_2_validator: Optional[BoundingBoxDoa] = None
    # data type: KernelBasedDoa
    oneof_schema_3_validator: Optional[KernelBasedDoa] = None
    # data type: MeanVarDoa
    oneof_schema_4_validator: Optional[MeanVarDoa] = None
    # data type: MahalanobisDoa
    oneof_schema_5_validator: Optional[MahalanobisDoa] = None
    # data type: CityBlockDoa
    oneof_schema_6_validator: Optional[CityBlockDoa] = None
    actual_instance: Optional[
        Union[
            BoundingBoxDoa,
            CityBlockDoa,
            KernelBasedDoa,
            LeverageDoa,
            MahalanobisDoa,
            MeanVarDoa,
        ]
    ] = None
    one_of_schemas: Set[str] = {
        "BoundingBoxDoa",
        "CityBlockDoa",
        "KernelBasedDoa",
        "LeverageDoa",
        "MahalanobisDoa",
        "MeanVarDoa",
    }

    model_config = ConfigDict(
        validate_assignment=True,
        protected_namespaces=(),
    )

    def __init__(self, *args, **kwargs) -> None:
        if args:
            if len(args) > 1:
                raise ValueError(
                    "If a position argument is used, only 1 is allowed to set `actual_instance`"
                )
            if kwargs:
                raise ValueError(
                    "If a position argument is used, keyword arguments cannot be used."
                )
            super().__init__(actual_instance=args[0])
        else:
            super().__init__(**kwargs)

    @field_validator("actual_instance")
    def actual_instance_must_validate_oneof(cls, v):
        instance = DoaData.model_construct()
        error_messages = []
        match = 0
        # validate data type: LeverageDoa
        if not isinstance(v, LeverageDoa):
            error_messages.append(f"Error! Input type `{type(v)}` is not `LeverageDoa`")
        else:
            match += 1
        # validate data type: BoundingBoxDoa
        if not isinstance(v, BoundingBoxDoa):
            error_messages.append(
                f"Error! Input type `{type(v)}` is not `BoundingBoxDoa`"
            )
        else:
            match += 1
        # validate data type: KernelBasedDoa
        if not isinstance(v, KernelBasedDoa):
            error_messages.append(
                f"Error! Input type `{type(v)}` is not `KernelBasedDoa`"
            )
        else:
            match += 1
        # validate data type: MeanVarDoa
        if not isinstance(v, MeanVarDoa):
            error_messages.append(f"Error! Input type `{type(v)}` is not `MeanVarDoa`")
        else:
            match += 1
        # validate data type: MahalanobisDoa
        if not isinstance(v, MahalanobisDoa):
            error_messages.append(
                f"Error! Input type `{type(v)}` is not `MahalanobisDoa`"
            )
        else:
            match += 1
        # validate data type: CityBlockDoa
        if not isinstance(v, CityBlockDoa):
            error_messages.append(
                f"Error! Input type `{type(v)}` is not `CityBlockDoa`"
            )
        else:
            match += 1
        if match > 1:
            # more than 1 match
            raise ValueError(
                "Multiple matches found when setting `actual_instance` in DoaData with oneOf schemas: BoundingBoxDoa, CityBlockDoa, KernelBasedDoa, LeverageDoa, MahalanobisDoa, MeanVarDoa. Details: "
                + ", ".join(error_messages)
            )
        elif match == 0:
            # no match
            raise ValueError(
                "No match found when setting `actual_instance` in DoaData with oneOf schemas: BoundingBoxDoa, CityBlockDoa, KernelBasedDoa, LeverageDoa, MahalanobisDoa, MeanVarDoa. Details: "
                + ", ".join(error_messages)
            )
        else:
            return v

    @classmethod
    def from_dict(cls, obj: Union[str, Dict[str, Any]]) -> Self:
        return cls.from_json(json.dumps(obj))

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Returns the object represented by the json string"""
        instance = cls.model_construct()
        error_messages = []
        match = 0

        # deserialize data into LeverageDoa
        try:
            instance.actual_instance = LeverageDoa.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into BoundingBoxDoa
        try:
            instance.actual_instance = BoundingBoxDoa.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into KernelBasedDoa
        try:
            instance.actual_instance = KernelBasedDoa.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into MeanVarDoa
        try:
            instance.actual_instance = MeanVarDoa.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into MahalanobisDoa
        try:
            instance.actual_instance = MahalanobisDoa.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into CityBlockDoa
        try:
            instance.actual_instance = CityBlockDoa.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))

        if match > 1:
            # more than 1 match
            raise ValueError(
                "Multiple matches found when deserializing the JSON string into DoaData with oneOf schemas: BoundingBoxDoa, CityBlockDoa, KernelBasedDoa, LeverageDoa, MahalanobisDoa, MeanVarDoa. Details: "
                + ", ".join(error_messages)
            )
        elif match == 0:
            # no match
            raise ValueError(
                "No match found when deserializing the JSON string into DoaData with oneOf schemas: BoundingBoxDoa, CityBlockDoa, KernelBasedDoa, LeverageDoa, MahalanobisDoa, MeanVarDoa. Details: "
                + ", ".join(error_messages)
            )
        else:
            return instance

    def to_json(self) -> str:
        """Returns the JSON representation of the actual instance"""
        if self.actual_instance is None:
            return "null"

        if hasattr(self.actual_instance, "to_json") and callable(
            self.actual_instance.to_json
        ):
            return self.actual_instance.to_json()
        else:
            return json.dumps(self.actual_instance)

    def to_dict(
        self,
    ) -> Optional[
        Union[
            Dict[str, Any],
            BoundingBoxDoa,
            CityBlockDoa,
            KernelBasedDoa,
            LeverageDoa,
            MahalanobisDoa,
            MeanVarDoa,
        ]
    ]:
        """Returns the dict representation of the actual instance"""
        if self.actual_instance is None:
            return None

        if hasattr(self.actual_instance, "to_dict") and callable(
            self.actual_instance.to_dict
        ):
            return self.actual_instance.to_dict()
        else:
            # primitive type
            return self.actual_instance

    def to_str(self) -> str:
        """Returns the string representation of the actual instance"""
        return pprint.pformat(self.model_dump())
