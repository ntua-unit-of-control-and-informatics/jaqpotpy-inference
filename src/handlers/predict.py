from ..entities.prediction_request import PredictionRequestPydantic
from ..helpers import model_decoder, json_to_predreq

def model_post_handler(request: PredictionRequestPydantic):
    """
    Handles the prediction request for a model.

    Args:
        request (PredictionRequestPydantic): The prediction request object.

    Returns:
        dict: The final prediction results.
    """
    model = model_decoder.decode(request.rawModel[0]) 

    if len(request.dataset['features']) == 1 and request.dataset['features'][0]['name'] == 'Smiles':
        data_entry_all = json_to_predreq.decode_smiles(request)
        _ = model(data_entry_all)
    elif len(request.dataset['features']) > 1 and any(feature['name'] == 'Smiles' for feature in request.dataset['features']):
        Smiles_input, data_entry_all = json_to_predreq.decode_smiles_external(request)
        _ = model(Smiles_input, data_entry_all)
    elif len(request.dataset['features']) > 1 and all(feature['name'] != 'Smiles' for feature in request.dataset['features']):
        data_entry_all = json_to_predreq.decode_only_external(request)
        model_prediction = model.predict(data_entry_all).tolist()
        if isinstance(model_prediction[0], list):
            results = {'Y': [item[i] for item in model_prediction] for i in range(len(model_prediction[0]))}
        results['AD'] = [None for _ in range(len(model_prediction))]
        results['Probabilities'] = [[] for _ in range(len(model_prediction))]
        final_all = {"predictions": [dict(zip(results, t)) for t in zip(*results.values())]}
        return final_all

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
