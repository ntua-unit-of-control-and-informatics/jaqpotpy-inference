import pandas as pd
import onnx
from onnxruntime import InferenceSession
from jaqpotpy.datasets import JaqpotpyDataset
from src.helpers.recreate_preprocessor import recreate_preprocessor
from jaqpotpy.doa.doa import Leverage


def predict_onnx(model, dataset: JaqpotpyDataset, request):
    sess = InferenceSession(model.SerializeToString())
    input_feed = {}
    for independent_feature in model.graph.input:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
            independent_feature.type.tensor_type.elem_type
        )

        if len(model.graph.input) == 1:
            input_feed[independent_feature.name] = dataset.X.values.astype(np_dtype)
        else:
            input_feed[independent_feature.name] = (
                dataset.X[independent_feature.name]
                .values.astype(np_dtype)
                .reshape(-1, 1)
            )
    if request.model["doas"]:
        doas_results = []
        input_df = pd.DataFrame(input_feed["input"])
        for _, data_instance in input_df.iterrows():
            doa_instance_prediction = {}
            for doa_data in request.model["doas"]:
                if doa_data["method"] == "LEVERAGE":
                    doa_method = Leverage()
                    doa_method.h_star = doa_data["doaData"]["hStar"]
                    doa_method.doa_matrix = doa_data["doaData"]["array"]
                doa_instance_prediction[doa_method.__name__] = doa_method.predict(
                    pd.DataFrame(data_instance.values.reshape(1, -1))
                )[0]
                doas_results.append(doa_instance_prediction)

    onnx_prediction = sess.run(None, input_feed)
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
            if (
                len(request.model["dependentFeatures"]) == 1
                and preprocessor_name != "LabelEncoder"
            ):
                onnx_prediction = preprocessor_recreated.inverse_transform(
                    onnx_prediction.reshape(-1, 1)
                )
            onnx_prediction = preprocessor_recreated.inverse_transform(onnx_prediction)

    if len(request.model["dependentFeatures"]) == 1:
        onnx_prediction = onnx_prediction.flatten()

    return onnx_prediction, doas_predictions


def predict_proba_onnx(model, dataset: JaqpotpyDataset, request):
    sess = InferenceSession(model.SerializeToString())
    input_feed = {}
    for independent_feature in model.graph.input:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
            independent_feature.type.tensor_type.elem_type
        )
        if len(model.graph.input) == 1:
            input_feed[independent_feature.name] = dataset.X.values.astype(np_dtype)
        else:
            input_feed[independent_feature.name] = (
                dataset.X[independent_feature.name]
                .values.astype(np_dtype)
                .reshape(-1, 1)
            )
    onnx_probs = sess.run(None, input_feed)
    probs_list = []
    for instance in onnx_probs[1]:
        rounded_instance = {k: round(v, 3) for k, v in instance.items()}
        if request.model["extraConfig"]["preprocessors"][0]["name"] == "LabelEncoder":
            labels = request.model["extraConfig"]["preprocessors"][0]["config"][
                "classes_"
            ]
            rounded_instance = {labels[k]: v for k, v in rounded_instance.items()}

        probs_list.append(rounded_instance)

    return probs_list
