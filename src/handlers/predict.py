import base64
import torch
import onnxruntime
import numpy as np
import torch.nn.functional as F
from ..entities.prediction_request import PredictionRequestPydantic
from ..helpers import model_decoder, json_to_predreq
from ..helpers.predict_methods import predict_onnx, predict_proba_onnx
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer

# import sys
# import os

# current_dir = os.path.dirname(__file__)
# software_dir = os.path.abspath(os.path.join(current_dir, '../../../../../JQP'))
# sys.path.append(software_dir)


def model_post_handler(request: PredictionRequestPydantic):
    model = model_decoder.decode(request.model["rawModel"])
    data_entry_all = json_to_predreq.decode(request)
    prediction = predict_onnx(model, data_entry_all, request)
    if request.model['task'].lower() == "classification":
        probabilities = predict_proba_onnx(model, data_entry_all, request)
    else:
        probabilities = [None for _ in range(len(prediction))]

    results = {}
    for i, feature in enumerate(request.model['dependentFeatures']):
        key = feature['key']
        if len(request.model['dependentFeatures']) == 1:
            values = [str(item) for item in prediction]
        else:
            values = [str(item) for item in prediction[:, i]]
        results[key] = values

    results["Probabilities"] = [str(prob) for prob in probabilities]
    results["AD"] = [None for _ in range(len(prediction))]

    final_all = {"predictions": [dict(zip(results, t)) for t in zip(*results.values())]}

    return final_all


def graph_post_handler(request: PredictionRequestPydantic):
    # Obtain the request info
    onnx_model = base64.b64decode(request.model["rawModel"])
    ort_session = onnxruntime.InferenceSession(onnx_model)
    feat_config = request.extraConfig["torchConfig"]["featurizerConfig"]
    # Load the featurizer
    featurizer = SmilesGraphFeaturizer()
    featurizer.load_dict(feat_config)
    # Sort the allowable sets
    featurizer.sort_allowable_sets()
    smiles = request.dataset["input"][0]["SMILES"]

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    data = featurizer.featurize(smiles)
    # ONNX Inference
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(data.x),
        ort_session.get_inputs()[1].name: to_numpy(data.edge_index),
        ort_session.get_inputs()[2].name: to_numpy(
            torch.zeros(data.x.shape[0], dtype=torch.int64)
        ),
    }
    ort_outs = torch.tensor(np.array(ort_session.run(None, ort_inputs)))

    return graph_binary_classification(request, ort_outs)


def graph_binary_classification(request: PredictionRequestPydantic, onnx_output):
    # Classification
    target_name = request.model["dependentFeatures"][0]["name"]
    probs = [F.sigmoid(onnx_output).squeeze().tolist()]
    preds = [int(prob > 0.5) for prob in probs]
    # UI Results
    results = {}
    results["Probabilities"] = [str(prob) for prob in probs]
    results[target_name] = [str(pred) for pred in preds]
    final_all = {"predictions": [dict(zip(results, t)) for t in zip(*results.values())]}
    print(final_all)

    return final_all
