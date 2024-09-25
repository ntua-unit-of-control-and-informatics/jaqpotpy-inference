from ..entities.prediction_request import PredictionRequestPydantic
from ..helpers import model_decoder, json_to_predreq
from ..helpers.predict_methods import predict_onnx, predict_proba_onnx


def sklearn_post_handler(request: PredictionRequestPydantic):
    model = model_decoder.decode(request.model["rawModel"])
    data_entry_all = json_to_predreq.decode(request)
    prediction = predict_onnx(model, data_entry_all, request)
    task = request.model["task"].lower()
    if task == "binary_classification" or task == "multiclass_classification":
        probabilities = predict_proba_onnx(model, data_entry_all, request)
    else:
        probabilities = [None for _ in range(len(prediction))]

    results = {}
    for i, feature in enumerate(request.model["dependentFeatures"]):
        key = feature["key"]
        values = [
            str(item[i]) if len(request.model["dependentFeatures"]) > 1 else str(item)
            for item in prediction
        ]
        results[key] = values

    results["Probabilities"] = [str(prob) for prob in probabilities]
    results["AD"] = [None for _ in range(len(prediction))]

    final_all = {"predictions": [dict(zip(results, t)) for t in zip(*results.values())]}

    return final_all
