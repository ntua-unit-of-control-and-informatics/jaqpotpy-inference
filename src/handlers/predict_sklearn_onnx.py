from jaqpotpy.api.openapi import PredictionRequest, PredictionResponse

from ..helpers import model_decoder, json_to_predreq
from ..helpers.predict_methods import predict_onnx
import numpy as np


def sklearn_onnx_post_handler(request: PredictionRequest) -> PredictionResponse:
    model = model_decoder.decode(request.model.raw_model)
    preprocessor = (
        model_decoder.decode(request.model.raw_preprocessor)
        if request.model.raw_preprocessor
        else None
    )
    data_entry_all, jaqpot_row_ids = json_to_predreq.decode(request)
    predicted_values, probabilities, doa_predictions = predict_onnx(
        model, preprocessor, data_entry_all, request
    )

    predictions = []
    for jaqpot_row_id in jaqpot_row_ids:
        if len(request.model.dependent_features) == 1:
            predicted_values = predicted_values.reshape(-1, 1)
        jaqpot_row_id = int(jaqpot_row_id)
        results = {
            feature.key: int(predicted_values[jaqpot_row_id, i])
            if isinstance(
                predicted_values[jaqpot_row_id, i],
                (np.int16, np.int32, np.int64, np.longlong),
            )
            else float(predicted_values[jaqpot_row_id, i])
            if isinstance(
                predicted_values[jaqpot_row_id, i], (np.float16, np.float32, np.float64)
            )
            else predicted_values[jaqpot_row_id, i]
            for i, feature in enumerate(request.model.dependent_features)
        }
        results["jaqpotMetadata"] = {
            "doa": doa_predictions[jaqpot_row_id] if doa_predictions else None,
            "probabilities": probabilities[jaqpot_row_id],
            "jaqpotRowId": jaqpot_row_id,
        }
        predictions.append(results)
    return PredictionResponse(predictions=predictions)
