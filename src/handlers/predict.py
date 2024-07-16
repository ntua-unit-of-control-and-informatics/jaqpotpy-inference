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

    results = {model.dependentFeatures[0]['name']: [str(item) for item in prediction]}
    results['Probabilities'] = [str(prob) for prob in probabilities]
    results['AD'] = [None for _ in range(len(prediction))]

    final_all = {"predictions": [dict(zip(results, t)) for t in zip(*results.values())]}

    return final_all