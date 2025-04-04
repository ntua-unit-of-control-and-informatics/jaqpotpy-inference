import base64
import io
import numpy as np
import onnx
import onnxruntime
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from jaqpot_api_client import PredictionRequest, PredictionResponse, FeatureType

from ..helpers.dataset_utils import build_dataset_from_request
from src.helpers.predict_methods import predict_onnx
from src.helpers.torch_utils import to_numpy


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

    data_entry_all, jaqpot_row_ids = build_dataset_from_request(request)
    predicted_values, probabilities, doa_predictions = predict_onnx(
        model, preprocessor, data_entry_all, request
    )

    predictions = []
    for jaqpot_row_id in jaqpot_row_ids:
        if len(request.model.dependent_features) == 1:
            predicted_values = predicted_values.reshape(-1, 1)
        jaqpot_row_id = int(jaqpot_row_id)
        results = {}
        for i, feature in enumerate(request.model.dependent_features):
            value = predicted_values[jaqpot_row_id, i]

            # Handle torch.Tensor
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()

            # Scalars
            if isinstance(value, (np.integer, int)):
                results[feature.key] = int(value)
            elif isinstance(value, (np.floating, float)):
                results[feature.key] = float(value)

            # Anything else (e.g., string, bool, object)
            else:
                results[feature.key] = value
        results["jaqpotMetadata"] = {
            "doa": doa_predictions[jaqpot_row_id] if doa_predictions else None,
            "probabilities": probabilities[jaqpot_row_id],
            "jaqpotRowId": jaqpot_row_id,
        }
        predictions.append(results)
    return PredictionResponse(predictions=predictions)

    # model_session = onnxruntime.InferenceSession(raw_model)
    # preprocessor_session = (
    #     onnxruntime.InferenceSession(base64.b64decode(raw_preprocessor))
    #     if has_preprocessor else None
    # )
    #
    # predictions = []
    # for i, row in enumerate(user_input):
    #     input_row = row.copy()
    #     for j, feature in enumerate(independent_features):
    #         if feature.feature_type == FeatureType.IMAGE:
    #             img = validate_and_decode_image(input_row[feature.key])
    #             img_array = np.array(img, dtype=np.uint8)  # [H, W, C]
    #             img_array = img_array.reshape(1, *img_array.shape)  # [1, H, W, C]
    #             input_row[feature.key] = img_array
    #
    #     del input_row['jaqpotRowId']
    #     if has_preprocessor:
    #         processed = preprocessor_session.run(None, input_row)
    #     else:
    #         processed = input_row
    #
    #     ort_inputs = processed
    #     ort_outs = model_session.run(None, ort_inputs)
    #
    #     predictions.append({
    #         "jaqpotRowId": str(i),
    #     })

    return PredictionResponse(predictions=predictions)


def validate_and_decode_image(b64_string):
    try:
        image_bytes = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Validate format, throws if not valid image
        img = Image.open(io.BytesIO(image_bytes))  # Reopen image to use
        return img
    except Exception as e:
        raise ValueError("Invalid image input") from e
