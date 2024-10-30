import base64
import onnxruntime
import torch
import io
import numpy as np
import torch.nn.functional as f
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer

from src.api.openapi import PredictionResponse
from src.api.openapi.models.prediction_request import PredictionRequest


def graph_post_handler(request: PredictionRequest) -> PredictionResponse:
    feat_config = request.extra_config["torchConfig"].featurizerConfig
    featurizer = _load_featurizer(feat_config)
    target_name = request.model.dependent_features[0].name
    model_task = request.model.task
    user_input = request.dataset.input
    raw_model = request.model.raw_model
    predictions = []
    if request.model.type == "TORCH_ONNX":
        for inp in user_input:
            model_output = onnx_post_handler(
                raw_model, featurizer.featurize(inp["SMILES"])
            )
            predictions.append(check_model_task(model_task, target_name, model_output, inp))
    elif request.model.type == "TORCHSCRIPT":
        for inp in user_input:
            model_output = torchscript_post_handler(
                raw_model, featurizer.featurize(inp["SMILES"])
            )
            predictions.append(check_model_task(model_task, target_name, model_output, inp))
    return PredictionResponse(predictions=predictions)


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


def graph_regression(target_name, output, inp):
    pred = [output.squeeze().tolist()]
    results = {"jaqpotMetadata": {"jaqpotRowId": inp["jaqpotRowId"]}}
    if "jaqpotRowLabel" in inp:
        results["jaqpotMetadata"]["jaqpotRowLabel"] = inp["jaqpotRowLabel"]
    results[target_name] = pred
    return results


def graph_binary_classification(target_name, output, inp):
    proba = f.sigmoid(output).squeeze().tolist()
    pred = int(proba > 0.5)
    # UI Results
    results = {"jaqpotMetadata": {
        "probabilities": [round((1 - proba), 3), round(proba, 3)],
        "jaqpotRowId": inp["jaqpotRowId"],
    }}
    if "jaqpotRowLabel" in inp:
        results["jaqpotMetadata"]["jaqpotRowLabel"] = inp["jaqpotRowLabel"]
    results[target_name] = pred
    return results


def check_model_task(model_task, target_name, out, row_id):
    if model_task == "BINARY_CLASSIFICATION":
        return graph_binary_classification(target_name, out, row_id)
    elif model_task == "REGRESSION":
        return graph_regression(target_name, out, row_id)
    else:
        raise ValueError(
            "Only BINARY_CLASSIFICATION and REGRESSION tasks are supported"
        )
