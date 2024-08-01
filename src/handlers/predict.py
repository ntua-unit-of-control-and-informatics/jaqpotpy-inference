from ..entities.prediction_request import PredictionRequestPydantic
from ..helpers import model_decoder, json_to_predreq


def model_post_handler(request: PredictionRequestPydantic):
    model = model_decoder.decode(request.model['rawModel'])
    data_entry_all = json_to_predreq.decode(request, model)
    prediction = model.predict_onnx(data_entry_all)
    if model.task == 'classification':
        probabilities = model.predict_proba_onnx(data_entry_all)
    else:
        probabilities = [None for _ in range(len(prediction))]

    results = {}
    for i, feature in enumerate(model.dependentFeatures):
        key = feature['key']
        if len(model.dependentFeatures) == 1:
            values = [str(item) for item in prediction]
        else:
            values = [str(item) for item in prediction[:, i]]
        results[key] = values

    results['Probabilities'] = [str(prob) for prob in probabilities]
    results['AD'] = [None for _ in range(len(prediction))]

    final_all = {"predictions": [dict(zip(results, t)) for t in zip(*results.values())]}

    return final_all
