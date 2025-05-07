import base64
import onnxruntime
import torch
import io
import numpy as np

from src.helpers.model_loader import retrieve_raw_model_from_request
from src.helpers.torch_utils import to_numpy, generate_prediction_response
from jaqpot_api_client import ModelType, PredictionRequest, PredictionResponse
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer
from jaqpot_api_client.models.model_task import ModelTask


def torch_geometric_post_handler(request: PredictionRequest) -> PredictionResponse:
    feat_config = request.model.torch_config
    featurizer = _load_featurizer(feat_config)
    target_name = request.model.dependent_features[0].name
    model_task = request.model.task
    user_input = request.dataset.input
    raw_model = retrieve_raw_model_from_request(request)
    predictions = []
    if request.model.type == ModelType.TORCH_GEOMETRIC_ONNX:
        for inp in user_input:
            model_output = torch_geometric_onnx_post_handler(
                raw_model, featurizer.featurize(inp["SMILES"])
            )
            predictions.append(
                generate_prediction_response(model_task, target_name, model_output, inp)
            )
    elif request.model.type == ModelType.TORCHSCRIPT:
        for inp in user_input:
            model_output = torchscript_post_handler(
                raw_model, featurizer.featurize(inp["SMILES"])
            )
            predictions.append(
                generate_prediction_response(model_task, target_name, model_output, inp)
            )
    return PredictionResponse(predictions=predictions)


def torch_geometric_onnx_post_handler(onnx_model, data):
    ort_session = onnxruntime.InferenceSession(onnx_model)
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(data.x),
        ort_session.get_inputs()[1].name: to_numpy(data.edge_index),
        ort_session.get_inputs()[2].name: to_numpy(
            torch.zeros(data.x.shape[0], dtype=torch.int64)
        ),
    }
    ort_outs = torch.tensor(np.array(ort_session.run(None, ort_inputs)))
    return ort_outs


def torchscript_post_handler(torchscript_model, data):
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


def _load_featurizer(config):
    featurizer = SmilesGraphFeaturizer()
    featurizer.load_dict(config)
    featurizer.sort_allowable_sets()
    return featurizer
