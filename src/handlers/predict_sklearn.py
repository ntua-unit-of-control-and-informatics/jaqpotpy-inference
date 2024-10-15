from ..entities.prediction_request import PredictionRequestPydantic
from ..helpers import model_decoder, json_to_predreq
from ..helpers.predict_methods import predict_onnx, predict_proba_onnx
import numpy as np

from jaqpotpy.doa.doa import Leverage


def sklearn_post_handler(request: PredictionRequestPydantic):
    model = model_decoder.decode(request.model["rawModel"])
    data_entry_all, JaqpotInternalId = json_to_predreq.decode(request)
    prediction, doas_predictions = predict_onnx(model, data_entry_all, request)
    task = request.model["task"].lower()
    if task == "binary_classification" or task == "multiclass_classification":
        probabilities = predict_proba_onnx(model, data_entry_all, request)
    else:
        probabilities = [None for _ in range(len(prediction))]

    final_all = []
    for jaqpot_id in JaqpotInternalId:
        if len(request.model["dependentFeatures"]) == 1:
            prediction = prediction.reshape(-1, 1)
        jaqpot_id = int(jaqpot_id)
        results = {
            feature["key"]: int(prediction[jaqpot_id, i])
            if isinstance(
                prediction[jaqpot_id, i], (np.int16, np.int32, np.int64, np.longlong)
            )
            else float(prediction[jaqpot_id, i])
            if isinstance(
                prediction[jaqpot_id, i], (np.float16, np.float32, np.float64)
            )
            else prediction[jaqpot_id, i]
            for i, feature in enumerate(request.model["dependentFeatures"])
        }
        results["jaqpotInternalMetadata"] = {
            "jaqpotInternalId": jaqpot_id,
            "AD": doas_predictions[jaqpot_id],
            "Probabilities": probabilities[jaqpot_id],
        }
        final_all.append(results)

    return {"predictions": final_all}
