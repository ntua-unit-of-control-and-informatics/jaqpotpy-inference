from ..entities.prediction_request import PredictionRequestPydantic
import base64
import onnxruntime
import torch
import io
import numpy as np
import torch.nn.functional as F
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer


def graph_post_handler(request: PredictionRequestPydantic):

    feat_config = request.extraConfig["torchConfig"]["featurizerConfig"]
    featurizer = _load_featurizer(feat_config)
    target_name = request.model["dependentFeatures"][0]["name"]
    model_task = request.model["task"]
    smiles = request.dataset["input"][0]["SMILES"]
    data = featurizer.featurize(smiles)
    raw_model = request.model["rawModel"]
    if request.model["type"] == "TORCH_ONNX":
        model_output = onnx_post_handler(raw_model, data)
        return check_model_task(model_task, target_name, model_output)
    elif request.model["type"] == "TORCHSCRIPT":
        model_output = torchscript_post_handler(raw_model, data)
        return check_model_task(model_task, target_name, model_output)


def onnx_post_handler(raw_model, data):
    onnx_model = base64.b64decode(raw_model)
    ort_session = onnxruntime.InferenceSession(onnx_model)
    ort_inputs = {
        ort_session.get_inputs()[0].name: _to_numpy(data.x),
        ort_session.get_inputs()[1].name: _to_numpy(data.edge_index),
        ort_session.get_inputs()[2].name: _to_numpy(
            torch.zeros(data.x.shape[0], dtype=torch.int64)
        ),
    }
    ort_outs = torch.tensor(np.array(ort_session.run(None, ort_inputs)))
    return ort_outs


def torchscript_post_handler(raw_model, data):
    torchscript_model = base64.b64decode(raw_model)
    model_buffer = io.BytesIO(torchscript_model)
    model_buffer.seek(0)
    torchscript_model = torch.jit.load(model_buffer)
    torchscript_model.eval()
    with torch.no_grad():
        if data.edge_attr.shape[1] == 0:
            out = torchscript_model(data.x, data.edge_index, data.batch)
        else:
            out = torchscript_model(data.x, data.edge_index, data.batch, data.edge_attr)
    return out


def _to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def _load_featurizer(config):

    featurizer = SmilesGraphFeaturizer()
    featurizer.load_dict(config)
    featurizer.sort_allowable_sets()
    return featurizer


def graph_regression(target_name, output):
    preds = [output.squeeze().tolist()]
    results = {}
    results[target_name] = [str(pred) for pred in preds]
    final_all = {"predictions": [dict(zip(results, t)) for t in zip(*results.values())]}
    return final_all


def graph_binary_classification(target_name, output):
    probs = [F.sigmoid(output).squeeze().tolist()]
    preds = [int(prob > 0.5) for prob in probs]
    # UI Results
    results = {}
    results["Probabilities"] = [str(prob) for prob in probs]
    results[target_name] = [str(pred) for pred in preds]
    final_all = {"predictions": [dict(zip(results, t)) for t in zip(*results.values())]}
    return final_all


def check_model_task(model_task, target_name, out):

    if model_task == "BINARY_CLASSIFICATION":
        return graph_binary_classification(target_name, out)
    elif model_task == "REGRESSION":
        return graph_regression(target_name, out)
    else:
        raise ValueError(
            "Only BINARY_CLASSIFICATION and REGRESSION tasks are supported"
        )
