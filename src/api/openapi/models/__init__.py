# coding: utf-8

# flake8: noqa
"""
Jaqpot API

A modern RESTful API for model management and prediction services, built using Spring Boot and Kotlin. Supports seamless integration with machine learning workflows.

The version of the OpenAPI document: 1.0.0
Contact: upci.ntua@gmail.com
Generated by OpenAPI Generator (https://openapi-generator.tech)

Do not edit the class manually.
"""  # noqa: E501

# import models into model package
from src.api.openapi.models.api_key import ApiKey
from src.api.openapi.models.binary_classification_scores import (
    BinaryClassificationScores,
)
from src.api.openapi.models.bounding_box_doa import BoundingBoxDoa
from src.api.openapi.models.city_block_doa import CityBlockDoa
from src.api.openapi.models.create_api_key201_response import CreateApiKey201Response
from src.api.openapi.models.create_invitations_request import CreateInvitationsRequest
from src.api.openapi.models.dataset import Dataset
from src.api.openapi.models.dataset_csv import DatasetCSV
from src.api.openapi.models.dataset_type import DatasetType
from src.api.openapi.models.doa import Doa
from src.api.openapi.models.doa_data import DoaData
from src.api.openapi.models.error_code import ErrorCode
from src.api.openapi.models.error_response import ErrorResponse
from src.api.openapi.models.feature import Feature
from src.api.openapi.models.feature_possible_value import FeaturePossibleValue
from src.api.openapi.models.feature_type import FeatureType
from src.api.openapi.models.get_all_api_keys_for_user200_response_inner import (
    GetAllApiKeysForUser200ResponseInner,
)
from src.api.openapi.models.get_datasets200_response import GetDatasets200Response
from src.api.openapi.models.get_models200_response import GetModels200Response
from src.api.openapi.models.kernel_based_doa import KernelBasedDoa
from src.api.openapi.models.lead import Lead
from src.api.openapi.models.leverage_doa import LeverageDoa
from src.api.openapi.models.library import Library
from src.api.openapi.models.mahalanobis_doa import MahalanobisDoa
from src.api.openapi.models.mean_var_doa import MeanVarDoa
from src.api.openapi.models.model import Model
from src.api.openapi.models.model_extra_config import ModelExtraConfig
from src.api.openapi.models.model_scores import ModelScores
from src.api.openapi.models.model_summary import ModelSummary
from src.api.openapi.models.model_task import ModelTask
from src.api.openapi.models.model_type import ModelType
from src.api.openapi.models.model_visibility import ModelVisibility
from src.api.openapi.models.multiclass_classification_scores import (
    MulticlassClassificationScores,
)
from src.api.openapi.models.organization import Organization
from src.api.openapi.models.organization_invitation import OrganizationInvitation
from src.api.openapi.models.organization_summary import OrganizationSummary
from src.api.openapi.models.organization_user import OrganizationUser
from src.api.openapi.models.organization_user_association_type import (
    OrganizationUserAssociationType,
)
from src.api.openapi.models.organization_visibility import OrganizationVisibility
from src.api.openapi.models.partial_update_organization_request import (
    PartialUpdateOrganizationRequest,
)
from src.api.openapi.models.partially_update_model_feature_request import (
    PartiallyUpdateModelFeatureRequest,
)
from src.api.openapi.models.partially_update_model_request import (
    PartiallyUpdateModelRequest,
)
from src.api.openapi.models.prediction_model import PredictionModel
from src.api.openapi.models.prediction_request import PredictionRequest
from src.api.openapi.models.prediction_response import PredictionResponse
from src.api.openapi.models.regression_scores import RegressionScores
from src.api.openapi.models.scores import Scores
from src.api.openapi.models.transformer import Transformer
from src.api.openapi.models.update_api_key200_response import UpdateApiKey200Response
from src.api.openapi.models.update_api_key_request import UpdateApiKeyRequest
from src.api.openapi.models.user import User
