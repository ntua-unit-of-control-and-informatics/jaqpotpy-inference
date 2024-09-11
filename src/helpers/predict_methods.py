from jaqpotpy.datasets.molecular_datasets import JaqpotpyDataset
from onnxruntime import InferenceSession
import numpy as np

def predict_onnx(model, dataset: JaqpotpyDataset, request):
    sess = InferenceSession(model.SerializeToString())
    onnx_prediction = sess.run(
        None, {"float_input": dataset.X.to_numpy().astype(np.float32)}
    )
    if len(request.model['dependentFeatures']) == 1:
        onnx_prediction[0] = onnx_prediction[0].reshape(-1, 1)
    return onnx_prediction[0]