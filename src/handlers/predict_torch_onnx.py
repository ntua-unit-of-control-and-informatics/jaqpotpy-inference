import base64
import numpy as np
import onnxruntime
import torch
from jaqpot_api_client import PredictionRequest, PredictionResponse
from src.helpers.torch_utils import to_numpy


def torch_onnx_post_handler(request: PredictionRequest) -> PredictionResponse:
    user_input = request.dataset.input
    raw_model = request.model.raw_model

    # Decode ONNX model and create one inference session per request
    onnx_model = base64.b64decode(raw_model)
    ort_session = onnxruntime.InferenceSession(onnx_model)

    predictions = []
    for i, inp in enumerate(user_input):
        output_tensor = onnx_infer(ort_session, inp)

        predictions.append(
            {"jaqpotRowId": str(i), "prediction": output_tensor.tolist()}
        )

    return PredictionResponse(predictions=predictions)


def onnx_infer(ort_session, data):
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
    ort_outs = ort_session.run(None, ort_inputs)
    return torch.tensor(np.array(ort_outs))  # batch dimension might be (1, N)
