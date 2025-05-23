import base64
import onnxruntime
import torch
import io
import numpy as np
import torch.nn.functional as f

from src.helpers.model_loader import retrieve_raw_model_from_request
from src.helpers.torch_utils import to_numpy, generate_prediction_response
from jaqpotpy.descriptors.tokenizer import SmilesVectorizer
from jaqpot_api_client import ModelType, PredictionRequest, PredictionResponse
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer


def torch_sequence_post_handler(request: PredictionRequest) -> PredictionResponse:
    feat_config = request.model.torch_config
    featurizer = _load_featurizer(feat_config)
    target_name = request.model.dependent_features[0].name
    model_task = request.model.task
    user_input = request.dataset.input
    raw_model = retrieve_raw_model_from_request(request)
    predictions = []
    for inp in user_input:
        model_output = onnx_post_handler(
            raw_model, featurizer.transform(featurizer.transform([inp["SMILES"]]))
        )
        predictions.append(
            generate_prediction_response(model_task, target_name, model_output, inp)
        )
    return PredictionResponse(predictions=predictions)


def onnx_post_handler(raw_model, data):
    onnx_model = base64.b64decode(raw_model)
    ort_session = onnxruntime.InferenceSession(onnx_model)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
    ort_outs = torch.tensor(np.array(ort_session.run(None, ort_inputs)))
    return ort_outs


def _load_featurizer(config):
    featurizer = SmilesVectorizer()
    featurizer.load_dict(config)
    return featurizer
