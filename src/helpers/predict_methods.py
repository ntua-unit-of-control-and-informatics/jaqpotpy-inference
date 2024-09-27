import numpy as np
import onnx
from onnxruntime import InferenceSession
from jaqpotpy.datasets import JaqpotpyDataset
from src.helpers.recreate_preprocessor import recreate_preprocessor


def predict_onnx(model, dataset: JaqpotpyDataset, request):
    sess = InferenceSession(model.SerializeToString())
    input_feed = {}
    for independent_feature in model.graph.input:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
            independent_feature.type.tensor_type.elem_type
        )
        if np_dtype in ["float32", "float64", "int32", "int64"]:
            target_dtype = np_dtype
        else:
            target_dtype = "object"
        if len(model.graph.input) == 1:
            input_feed[independent_feature.name] = dataset.X.values.astype(target_dtype)
        else:
            input_feed[independent_feature.name] = (
                dataset.X[independent_feature.name]
                .values.astype(target_dtype)
                .reshape(-1, 1)
            )
    onnx_prediction = sess.run(None, input_feed)
    if len(request.model["dependentFeatures"]) == 1:
        onnx_prediction = onnx_prediction[0].reshape(-1, 1)
        # onnx_prediction is being reshaped to a 2D array to avoid errors
        # when the model has only one dependent feature. In multi-output models,
        # onnx_prediction is already a 2D array.
    else:
        onnx_prediction = onnx_prediction[0]

    if request.model["extraConfig"]["preprocessors"]:
        for i in reversed(range(len(request.model["extraConfig"]["preprocessors"]))):
            preprocessor_name = request.model["extraConfig"]["preprocessors"][i]["name"]
            preprocessor_config = request.model["extraConfig"]["preprocessors"][i][
                "config"
            ]
            preprocessor_recreated = recreate_preprocessor(
                preprocessor_name, preprocessor_config
            )
            onnx_prediction = preprocessor_recreated.inverse_transform(onnx_prediction)

    if len(request.model["dependentFeatures"]) == 1:
        onnx_prediction = onnx_prediction.flatten()

    return onnx_prediction


def predict_proba_onnx(model, dataset: JaqpotpyDataset, request):
    sess = InferenceSession(model.SerializeToString())
    input_feed = {}
    for independent_feature in model.graph.input:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
            independent_feature.type.tensor_type.elem_type
        )
        if np_dtype in ["float32", "float64", "int32", "int64"]:
            target_dtype = np_dtype
        else:
            target_dtype = "object"
        if len(model.graph.input) == 1:
            input_feed[independent_feature.name] = dataset.X.values.astype(target_dtype)
        else:
            input_feed[independent_feature.name] = (
                dataset.X[independent_feature.name]
                .values.astype(target_dtype)
                .reshape(-1, 1)
            )
    onnx_probs = sess.run(None, input_feed)
    onnx_probs_list = [
        max(onnx_probs[1][instance].values()) for instance in range(len(onnx_probs[1]))
    ]
    return onnx_probs_list
