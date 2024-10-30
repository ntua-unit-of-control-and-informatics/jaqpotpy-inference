# coding: utf-8

"""
Jaqpot API

A modern RESTful API for model management and prediction services, built using Spring Boot and Kotlin. Supports seamless integration with machine learning workflows.

The version of the OpenAPI document: 1.0.0
Contact: upci.ntua@gmail.com
Generated by OpenAPI Generator (https://openapi-generator.tech)

Do not edit the class manually.
"""  # noqa: E501

import warnings
from pydantic import validate_call, Field, StrictFloat, StrictStr, StrictInt
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Annotated

from pydantic import Field, StrictInt
from typing_extensions import Annotated
from src.api.openapi.models.feature import Feature
from src.api.openapi.models.partially_update_model_feature_request import (
    PartiallyUpdateModelFeatureRequest,
)

from src.api.openapi.api_client import ApiClient, RequestSerialized
from src.api.openapi.api_response import ApiResponse
from src.api.openapi.rest import RESTResponseType


class FeatureApi:
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient.get_default()
        self.api_client = api_client

    @validate_call
    def partially_update_model_feature(
        self,
        model_id: Annotated[
            StrictInt, Field(description="The ID of the model containing the feature")
        ],
        feature_id: Annotated[
            StrictInt, Field(description="The ID of the feature to update")
        ],
        partially_update_model_feature_request: PartiallyUpdateModelFeatureRequest,
        _request_timeout: Union[
            None,
            Annotated[StrictFloat, Field(gt=0)],
            Tuple[
                Annotated[StrictFloat, Field(gt=0)], Annotated[StrictFloat, Field(gt=0)]
            ],
        ] = None,
        _request_auth: Optional[Dict[StrictStr, Any]] = None,
        _content_type: Optional[StrictStr] = None,
        _headers: Optional[Dict[StrictStr, Any]] = None,
        _host_index: Annotated[StrictInt, Field(ge=0, le=0)] = 0,
    ) -> Feature:
        """Update a feature for a specific model

        Update the name, description, and feature type of an existing feature within a specific model

        :param model_id: The ID of the model containing the feature (required)
        :type model_id: int
        :param feature_id: The ID of the feature to update (required)
        :type feature_id: int
        :param partially_update_model_feature_request: (required)
        :type partially_update_model_feature_request: PartiallyUpdateModelFeatureRequest
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :type _request_timeout: int, tuple(int, int), optional
        :param _request_auth: set to override the auth_settings for an a single
                              request; this effectively ignores the
                              authentication in the spec for a single request.
        :type _request_auth: dict, optional
        :param _content_type: force content-type for the request.
        :type _content_type: str, Optional
        :param _headers: set to override the headers for a single
                         request; this effectively ignores the headers
                         in the spec for a single request.
        :type _headers: dict, optional
        :param _host_index: set to override the host_index for a single
                            request; this effectively ignores the host_index
                            in the spec for a single request.
        :type _host_index: int, optional
        :return: Returns the result object.
        """  # noqa: E501

        _param = self._partially_update_model_feature_serialize(
            model_id=model_id,
            feature_id=feature_id,
            partially_update_model_feature_request=partially_update_model_feature_request,
            _request_auth=_request_auth,
            _content_type=_content_type,
            _headers=_headers,
            _host_index=_host_index,
        )

        _response_types_map: Dict[str, Optional[str]] = {
            "200": "Feature",
            "400": None,
            "404": None,
            "401": None,
            "403": None,
        }
        response_data = self.api_client.call_api(
            *_param, _request_timeout=_request_timeout
        )
        response_data.read()
        return self.api_client.response_deserialize(
            response_data=response_data,
            response_types_map=_response_types_map,
        ).data

    @validate_call
    def partially_update_model_feature_with_http_info(
        self,
        model_id: Annotated[
            StrictInt, Field(description="The ID of the model containing the feature")
        ],
        feature_id: Annotated[
            StrictInt, Field(description="The ID of the feature to update")
        ],
        partially_update_model_feature_request: PartiallyUpdateModelFeatureRequest,
        _request_timeout: Union[
            None,
            Annotated[StrictFloat, Field(gt=0)],
            Tuple[
                Annotated[StrictFloat, Field(gt=0)], Annotated[StrictFloat, Field(gt=0)]
            ],
        ] = None,
        _request_auth: Optional[Dict[StrictStr, Any]] = None,
        _content_type: Optional[StrictStr] = None,
        _headers: Optional[Dict[StrictStr, Any]] = None,
        _host_index: Annotated[StrictInt, Field(ge=0, le=0)] = 0,
    ) -> ApiResponse[Feature]:
        """Update a feature for a specific model

        Update the name, description, and feature type of an existing feature within a specific model

        :param model_id: The ID of the model containing the feature (required)
        :type model_id: int
        :param feature_id: The ID of the feature to update (required)
        :type feature_id: int
        :param partially_update_model_feature_request: (required)
        :type partially_update_model_feature_request: PartiallyUpdateModelFeatureRequest
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :type _request_timeout: int, tuple(int, int), optional
        :param _request_auth: set to override the auth_settings for an a single
                              request; this effectively ignores the
                              authentication in the spec for a single request.
        :type _request_auth: dict, optional
        :param _content_type: force content-type for the request.
        :type _content_type: str, Optional
        :param _headers: set to override the headers for a single
                         request; this effectively ignores the headers
                         in the spec for a single request.
        :type _headers: dict, optional
        :param _host_index: set to override the host_index for a single
                            request; this effectively ignores the host_index
                            in the spec for a single request.
        :type _host_index: int, optional
        :return: Returns the result object.
        """  # noqa: E501

        _param = self._partially_update_model_feature_serialize(
            model_id=model_id,
            feature_id=feature_id,
            partially_update_model_feature_request=partially_update_model_feature_request,
            _request_auth=_request_auth,
            _content_type=_content_type,
            _headers=_headers,
            _host_index=_host_index,
        )

        _response_types_map: Dict[str, Optional[str]] = {
            "200": "Feature",
            "400": None,
            "404": None,
            "401": None,
            "403": None,
        }
        response_data = self.api_client.call_api(
            *_param, _request_timeout=_request_timeout
        )
        response_data.read()
        return self.api_client.response_deserialize(
            response_data=response_data,
            response_types_map=_response_types_map,
        )

    @validate_call
    def partially_update_model_feature_without_preload_content(
        self,
        model_id: Annotated[
            StrictInt, Field(description="The ID of the model containing the feature")
        ],
        feature_id: Annotated[
            StrictInt, Field(description="The ID of the feature to update")
        ],
        partially_update_model_feature_request: PartiallyUpdateModelFeatureRequest,
        _request_timeout: Union[
            None,
            Annotated[StrictFloat, Field(gt=0)],
            Tuple[
                Annotated[StrictFloat, Field(gt=0)], Annotated[StrictFloat, Field(gt=0)]
            ],
        ] = None,
        _request_auth: Optional[Dict[StrictStr, Any]] = None,
        _content_type: Optional[StrictStr] = None,
        _headers: Optional[Dict[StrictStr, Any]] = None,
        _host_index: Annotated[StrictInt, Field(ge=0, le=0)] = 0,
    ) -> RESTResponseType:
        """Update a feature for a specific model

        Update the name, description, and feature type of an existing feature within a specific model

        :param model_id: The ID of the model containing the feature (required)
        :type model_id: int
        :param feature_id: The ID of the feature to update (required)
        :type feature_id: int
        :param partially_update_model_feature_request: (required)
        :type partially_update_model_feature_request: PartiallyUpdateModelFeatureRequest
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :type _request_timeout: int, tuple(int, int), optional
        :param _request_auth: set to override the auth_settings for an a single
                              request; this effectively ignores the
                              authentication in the spec for a single request.
        :type _request_auth: dict, optional
        :param _content_type: force content-type for the request.
        :type _content_type: str, Optional
        :param _headers: set to override the headers for a single
                         request; this effectively ignores the headers
                         in the spec for a single request.
        :type _headers: dict, optional
        :param _host_index: set to override the host_index for a single
                            request; this effectively ignores the host_index
                            in the spec for a single request.
        :type _host_index: int, optional
        :return: Returns the result object.
        """  # noqa: E501

        _param = self._partially_update_model_feature_serialize(
            model_id=model_id,
            feature_id=feature_id,
            partially_update_model_feature_request=partially_update_model_feature_request,
            _request_auth=_request_auth,
            _content_type=_content_type,
            _headers=_headers,
            _host_index=_host_index,
        )

        _response_types_map: Dict[str, Optional[str]] = {
            "200": "Feature",
            "400": None,
            "404": None,
            "401": None,
            "403": None,
        }
        response_data = self.api_client.call_api(
            *_param, _request_timeout=_request_timeout
        )
        return response_data.response

    def _partially_update_model_feature_serialize(
        self,
        model_id,
        feature_id,
        partially_update_model_feature_request,
        _request_auth,
        _content_type,
        _headers,
        _host_index,
    ) -> RequestSerialized:
        _host = None

        _collection_formats: Dict[str, str] = {}

        _path_params: Dict[str, str] = {}
        _query_params: List[Tuple[str, str]] = []
        _header_params: Dict[str, Optional[str]] = _headers or {}
        _form_params: List[Tuple[str, str]] = []
        _files: Dict[
            str, Union[str, bytes, List[str], List[bytes], List[Tuple[str, bytes]]]
        ] = {}
        _body_params: Optional[bytes] = None

        # process the path parameters
        if model_id is not None:
            _path_params["modelId"] = model_id
        if feature_id is not None:
            _path_params["featureId"] = feature_id
        # process the query parameters
        # process the header parameters
        # process the form parameters
        # process the body parameter
        if partially_update_model_feature_request is not None:
            _body_params = partially_update_model_feature_request

        # set the HTTP header `Accept`
        if "Accept" not in _header_params:
            _header_params["Accept"] = self.api_client.select_header_accept(
                ["application/json"]
            )

        # set the HTTP header `Content-Type`
        if _content_type:
            _header_params["Content-Type"] = _content_type
        else:
            _default_content_type = self.api_client.select_header_content_type(
                ["application/json"]
            )
            if _default_content_type is not None:
                _header_params["Content-Type"] = _default_content_type

        # authentication setting
        _auth_settings: List[str] = ["bearerAuth"]

        return self.api_client.param_serialize(
            method="PATCH",
            resource_path="/v1/models/{modelId}/features/{featureId}",
            path_params=_path_params,
            query_params=_query_params,
            header_params=_header_params,
            body=_body_params,
            post_params=_form_params,
            files=_files,
            auth_settings=_auth_settings,
            collection_formats=_collection_formats,
            _host=_host,
            _request_auth=_request_auth,
        )