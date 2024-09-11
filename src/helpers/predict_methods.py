from jaqpotpy.datasets.molecular_datasets import JaqpotpyDataset
from src.helpers.recreate_preprocessor import recreate_preprocessor
from onnxruntime import InferenceSession
import numpy as np

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


    return onnx_prediction[0]