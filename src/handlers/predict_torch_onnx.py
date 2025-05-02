import base64

import numpy as np
import onnx
import torch
from jaqpot_api_client import PredictionRequest, PredictionResponse, FeatureType

from src.helpers.predict_methods import predict_torch_onnx
from ..helpers.dataset_utils import build_tensor_dataset_from_request
from ..helpers.image_utils import validate_and_decode_image, tensor_to_base64_img
from ..s3_client import ModelS3Client


s3_client = ModelS3Client()


def convert_tensor_to_base64_image(image_array: torch.Tensor) -> str:
    """
    Converts a tensor of shape [C, H, W] to base64-encoded PNG.
    Supports grayscale (1 channel) or RGB (3 channels).
    """
    if isinstance(image_array, np.ndarray):
        image_array = torch.tensor(image_array)

    if image_array.ndim == 3 and image_array.shape[0] in [1, 3]:
        return tensor_to_base64_img(image_array)

    raise ValueError("Expected tensor of shape [C, H, W] with 1 or 3 channels")


def torch_onnx_post_handler(request: PredictionRequest) -> PredictionResponse:
    if request.model.raw_model is None:
        raw_model = s3_client.download_file(request.model.id)
        model = onnx.load(raw_model)
    else:
        raw_model = base64.b64decode(request.model.raw_model)
        model = onnx.load_from_string(raw_model)

    preprocessor = (
        onnx.load_from_string(base64.b64decode(request.model.raw_preprocessor))
        if request.model.raw_preprocessor
        else None
    )

    # Find image features
    image_features = [
        f
        for f in request.model.independent_features
        if f.feature_type == FeatureType.IMAGE
    ]

    # Decode all images in input
    for row in request.dataset.input:
        for f in image_features:
            pil_img = validate_and_decode_image(row[f.key])  # from base64
            np_img = np.array(pil_img)  # shape [H, W] or [H, W, C]

            # Ensure [H, W, C] with 3 channels
            if np_img.ndim == 2:
                np_img = np.expand_dims(np_img, axis=-1)  # [H, W, 1]

            if np_img.shape[2] not in [1, 3]:
                raise ValueError("Only 1 or 3 channel images are supported")

            # Store the numpy image (ready to convert to torch later)
            row[f.key] = np_img

    # Build dataset and run prediction
    data_entry_all, jaqpot_row_ids = build_tensor_dataset_from_request(request)
    predicted_values = predict_torch_onnx(model, preprocessor, data_entry_all, request)

    predictions = []
    for jaqpot_row_id in jaqpot_row_ids:
        jaqpot_row_id = int(jaqpot_row_id)
        results = {}

        value = predicted_values[jaqpot_row_id]

        for i, feature in enumerate(request.model.dependent_features):
            if isinstance(value, (np.ndarray, torch.Tensor)):
                tensor = torch.tensor(value) if isinstance(value, np.ndarray) else value

                if (
                    request.dataset.result_types is not None
                    and request.dataset.result_types.get(feature.key)
                    and feature.feature_type == FeatureType.IMAGE
                ):
                    if tensor.ndim == 4:  # remove batch dim if present
                        tensor = tensor.squeeze(0)

                    if tensor.ndim == 3:
                        results[feature.key] = convert_tensor_to_base64_image(tensor)
                    else:
                        raise ValueError("Unexpected image tensor shape for output")
                else:
                    results[feature.key] = tensor.detach().cpu().numpy().tolist()
            elif isinstance(value, (np.integer, int)):
                results[feature.key] = int(value)
            elif isinstance(value, (np.floating, float)):
                results[feature.key] = float(value)
            else:
                results[feature.key] = value

        results["jaqpotMetadata"] = {"jaqpotRowId": jaqpot_row_id}
        predictions.append(results)

    return PredictionResponse(predictions=predictions)
