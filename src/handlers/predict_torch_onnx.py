import base64
import numpy as np
import onnx
import torch
from jaqpot_api_client import PredictionRequest, PredictionResponse, FeatureType

from ..helpers.dataset_utils import (
    build_tensor_dataset_from_request,
)
from src.helpers.predict_methods import predict_torch_onnx
from ..helpers.image_utils import validate_and_decode_image, tensor_to_base64_img


def convert_tensor_to_base64_image(tensor: np.ndarray) -> str:
    return tensor_to_base64_img(torch.tensor(tensor))


def torch_onnx_post_handler(request: PredictionRequest) -> PredictionResponse:
    model = onnx.load_from_string(base64.b64decode(request.model.raw_model))
    preprocessor = (
        onnx.load_from_string(base64.b64decode(request.model.raw_preprocessor))
        if request.model.raw_preprocessor
        else None
    )

    image_features = [
        f
        for f in request.model.independent_features
        if f.feature_type == FeatureType.IMAGE
    ]

    for row in request.dataset.input:
        for f in image_features:
            img = validate_and_decode_image(row[f.key])
            img_array = np.array(img, dtype=np.uint8)  # [H, W, C]
            img_array = img_array.reshape(1, *img_array.shape)  # [1, H, W, C]
            row[f.key] = img_array

    data_entry_all, jaqpot_row_ids = build_tensor_dataset_from_request(request)
    predicted_values = predict_torch_onnx(model, preprocessor, data_entry_all, request)

    predictions = []
    for jaqpot_row_id in jaqpot_row_ids:
        jaqpot_row_id = int(jaqpot_row_id)
        results = {}
        for i, feature in enumerate(request.model.dependent_features):
            value = predicted_values[jaqpot_row_id]

            if isinstance(value, np.ndarray):
                if (
                    request.dataset.result_types[feature.key]
                    and feature.feature_type == FeatureType.IMAGE
                ):
                    results[feature.key] = convert_tensor_to_base64_image(value)
                else:
                    results[feature.key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                results[feature.key] = value.detach().cpu().numpy()
            elif isinstance(value, (np.integer, int)):
                results[feature.key] = int(value)
            elif isinstance(value, (np.floating, float)):
                results[feature.key] = float(value)
            # Anything else (e.g., string, bool, object)
            else:
                results[feature.key] = value
        results["jaqpotMetadata"] = {
            "jaqpotRowId": jaqpot_row_id,
        }
        predictions.append(results)
    return PredictionResponse(predictions=predictions)
