import numpy as np
from onnxruntime import InferenceSession
from jaqpotpy.datasets.molecular_datasets import JaqpotpyDataset
from src.helpers.recreate_preprocessor import recreate_preprocessor


def predict_onnx(model, dataset: JaqpotpyDataset, request):
    sess = InferenceSession(model.SerializeToString())
    onnx_prediction = sess.run(
        None, {"float_input": dataset.X.to_numpy().astype(np.float32)}
    )
    if len(request.model['dependentFeatures']) == 1:
        onnx_prediction = onnx_prediction[0].reshape(-1, 1)

    if request.model['extraConfig']['preprocessors']:
        for i in reversed(range(len(request.model['extraConfig']['preprocessors']))):
            preprocessor_name = request.model['extraConfig']['preprocessors'][i]['name']
            preprocessor_config = request.model['extraConfig']['preprocessors'][i]['config']
            preprocessor_recreated = recreate_preprocessor(preprocessor_name, preprocessor_config)
            onnx_prediction = preprocessor_recreated.inverse_transform(onnx_prediction)

    if len(request.model['dependentFeatures']) == 1:
        onnx_prediction = onnx_prediction.flatten()
    return onnx_prediction

def predict_proba_onnx(model, dataset: JaqpotpyDataset, request):
    sess = InferenceSession(model.SerializeToString())
    onnx_probs = sess.run(
        None, {"float_input": dataset.X.to_numpy().astype(np.float32)}
    )
    onnx_probs_list = [
        max(onnx_probs[1][instance].values())
        for instance in range(len(onnx_probs[1]))
    ]
    return onnx_probs_list