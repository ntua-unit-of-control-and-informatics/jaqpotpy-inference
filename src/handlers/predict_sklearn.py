from ..api.openapi import PredictionResponse
from ..api.openapi.models.prediction_request import PredictionRequest
from ..helpers import model_decoder, json_to_predreq
from ..helpers.predict_methods import predict_onnx, predict_proba_onnx
import numpy as np


def sklearn_post_handler(request: PredictionRequest) -> PredictionResponse:
    model = model_decoder.decode(request.model.raw_model)
    preprocessor = (
        model_decoder.decode(request.model.raw_preprocessor)
        if request.model.raw_preprocessor
        else None
    )
    data_entry_all, jaqpot_row_ids = json_to_predreq.decode(request)
    prediction, doa_predictions = predict_onnx(
        model, preprocessor, data_entry_all, request
    )
    task = request.model.task.lower()
    if task == "binary_classification" or task == "multiclass_classification":
        probabilities = predict_proba_onnx(model, data_entry_all, request)
    else:
        probabilities = [None for _ in range(len(prediction))]

    predictions = []
    for jaqpot_row_id in jaqpot_row_ids:
        if len(request.model.dependent_features) == 1:
            prediction = prediction.reshape(-1, 1)
        jaqpot_row_id = int(jaqpot_row_id)
        results = {
            feature.key: int(prediction[jaqpot_row_id, i])
            if isinstance(
                prediction[jaqpot_row_id, i],
                (np.int16, np.int32, np.int64, np.longlong),
            )
            else float(prediction[jaqpot_row_id, i])
            if isinstance(
                prediction[jaqpot_row_id, i], (np.float16, np.float32, np.float64)
            )
            else prediction[jaqpot_row_id, i]
            for i, feature in enumerate(request.model.dependent_features)
        }
        results["jaqpotMetadata"] = {
            "doa": doa_predictions[jaqpot_row_id] if doa_predictions else None,
            "probabilities": probabilities[jaqpot_row_id],
            "jaqpotRowId": jaqpot_row_id,
        }
        predictions.append(results)
    return PredictionResponse(predictions=predictions)
