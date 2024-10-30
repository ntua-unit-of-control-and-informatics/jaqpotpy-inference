import pandas as pd
import onnx
from onnxruntime import InferenceSession
from jaqpotpy.datasets import JaqpotpyDataset
from src.helpers.recreate_preprocessor import recreate_preprocessor
from jaqpotpy.doa import Leverage, BoundingBox, MeanVar


def calculate_doas(input_feed, request):
    """
    Calculate the Domain of Applicability (DoA) for given input data using specified methods.
    Args:
        input_feed (dict): A dictionary containing the input data under the key "input".
        request (object): An object containing the model information, specifically the DoA methods
                          and their corresponding data under the key "model".
    Returns:
        list: A list of dictionaries where each dictionary contains the DoA predictions for a single
              data instance. The keys in the dictionary are the names of the DoA methods used, and
              the values are the corresponding DoA predictions.
    """

    doas_results = []
    input_df = pd.DataFrame(input_feed["input"])
    for _, data_instance in input_df.iterrows():
        doa_instance_prediction = {}
        for doa_data in request.model.doas:
            if doa_data.method == "LEVERAGE":
                doa_method = Leverage()
                doa_method.h_star = doa_data.data.hStar
                doa_method.doa_matrix = doa_data.data.doaMatrix
            elif doa_data.method == "BOUNDING_BOX":
                doa_method = BoundingBox()
                doa_method.bounding_box = doa_data.data.boundingBox
            elif doa_data.method == "MEAN_VAR":
                doa_method = MeanVar()
                doa_method.bounds = doa_data.data.bounds
            doa_instance_prediction[doa_method.__name__] = doa_method.predict(
                pd.DataFrame(data_instance.values.reshape(1, -1))
            )[0]
        # Majority voting
        if len(request.model.doas) > 1:
            in_doa_values = [
                value["inDoa"] for value in doa_instance_prediction.values()
            ]
            doa_instance_prediction["majorityVoting"] = in_doa_values.count(True) > (
                len(in_doa_values) / 2
            )
        else:
            doa_instance_prediction["majorityVoting"] = None
        doas_results.append(doa_instance_prediction)
    return doas_results


def predict_onnx(model, dataset: JaqpotpyDataset, request):
    """
    Perform prediction using an ONNX model.
    Parameters:
    model (onnx.ModelProto): The ONNX model to be used for prediction.
    dataset (JaqpotpyDataset): The dataset containing the input features.
    request (dict): A dictionary containing additional configuration for the prediction,
                    including model-specific settings and preprocessors.
    Returns:
    tuple: A tuple containing the ONNX model predictions and DOA results (if applicable).
    The function performs the following steps:
    1. Initializes an ONNX InferenceSession with the serialized model.
    2. Prepares the input feed by converting dataset features to the appropriate numpy data types.
    3. If doas (Domain of Applicability) is requested, it calculates the DOAS results.
    4. Runs the ONNX model to get predictions.
    5. Applies any specified preprocessors in reverse order to the predictions.
    6. Flattens the predictions if there is only one dependent feature.
    Note:
    - The function assumes that the dataset features and model inputs are aligned.
    - The request dictionary should contain the necessary configuration for preprocessors and DOAS.
    """

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
    if request.model.doas:
        doas_results = calculate_doas(input_feed, request)
    else:
        doas_results = None

    onnx_prediction = sess.run(None, input_feed)[0]

    if request.model.extra_config["preprocessors"]:
        for i in reversed(range(len(request.model.extra_config["preprocessors"]))):
            preprocessor_name = request.model.extra_config["preprocessors"][i].name
            preprocessor_config = request.model.extra_config["preprocessors"][i].config
            preprocessor_recreated = recreate_preprocessor(
                preprocessor_name, preprocessor_config
            )
            if (
                len(request.model.dependent_features) == 1
                and preprocessor_name != "LabelEncoder"
            ):
                onnx_prediction = preprocessor_recreated.inverse_transform(
                    onnx_prediction.reshape(-1, 1)
                )
            onnx_prediction = preprocessor_recreated.inverse_transform(onnx_prediction)

    if len(request.model.dependent_features) == 1:
        onnx_prediction = onnx_prediction.flatten()

    return onnx_prediction, doas_results


def predict_proba_onnx(model, dataset: JaqpotpyDataset, request):
    """
    Predict the probability estimates for a given dataset using an ONNX model.
    Parameters:
    model (onnx.ModelProto): The ONNX model used for prediction.
    dataset (JaqpotpyDataset): The dataset containing the features for prediction.
    request (dict): A dictionary containing additional request information, including model configuration.
    Returns:
    list: A list of dictionaries where each dictionary contains the predicted probabilities for each class,
          with class labels as keys and rounded probability values as values.
    """

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
        if (
            request.model.extra_config["preprocessors"]
            and request.model.extra_config["preprocessors"][0].name
            == "LabelEncoder"
        ):
            labels = request.model.extra_config["preprocessors"][0].config[
                "classes_"
            ]
            rounded_instance = {labels[k]: v for k, v in rounded_instance.items()}

        probs_list.append(rounded_instance)

    return probs_list
