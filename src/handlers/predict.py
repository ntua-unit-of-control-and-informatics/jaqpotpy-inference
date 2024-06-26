from ..entities.prediction_request import PredictionRequestPydantic
from ..helpers import model_decoder, json_to_predreq


def model_post_handler(request: PredictionRequestPydantic):
    model = model_decoder.decode(request.model['actualModel'])
    data_entry_all = json_to_predreq.decode(request)
    _ = model(data_entry_all)

    if isinstance(model.prediction[0], list):
        results = {model.Y[i]: [item[i] for item in model.prediction] for i in range(len(model.prediction[0]))}
    elif isinstance(model.prediction, list):
        if isinstance(model.Y, list):
            results = {model.Y[0]: [item for item in model.prediction]}
        else:
            results = {model.Y: [item for item in model.prediction]}
    else:
        results = {model.Y: [item for item in model.prediction]}

    if model.doa:
        results['AD'] = model.doa.IN
    else:
        results['AD'] = [None for _ in range(len(model.prediction))]

    if model.probability:
        results['Probabilities'] = [list(prob) for prob in model.probability]
    else:
        results['Probabilities'] = [[] for _ in range(len(model.prediction))]

    final_all = {"predictions": [dict(zip(results, t)) for t in zip(*results.values())]}

    return final_all
