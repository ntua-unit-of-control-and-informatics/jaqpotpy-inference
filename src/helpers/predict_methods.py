import pandas as pd
import onnx
from onnxruntime import InferenceSession
from jaqpotpy.datasets import JaqpotpyDataset
from src.helpers.recreate_preprocessor import recreate_preprocessor
from jaqpotpy.doa import (
    Leverage,
    BoundingBox,
    MeanVar,
    Mahalanobis,
    KernelBased,
    CityBlock,
)


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
                doa_method.h_star = doa_data.data["hStar"]
                doa_method.doa_matrix = doa_data.data["doaMatrix"]
            elif doa_data.method == "BOUNDING_BOX":
                doa_method = BoundingBox()
                doa_method.bounding_box = doa_data.data["boundingBox"]
            elif doa_data.method == "MEAN_VAR":
                doa_method = MeanVar()
                doa_method.bounds = doa_data.data["bounds"]
            elif doa_data.method == "MAHALANOBIS":
                doa_method = Mahalanobis()
                doa_method._mean_vector = doa_data.data["meanVector"]
                doa_method._inv_cov_matrix = doa_data.data["covMatrix"]
                doa_method._threshold = doa_data.data["threshold"]
            elif doa_data.method == "KERNEL_BASED":
                doa_method = KernelBased()
                doa_method._sigma = doa_data.data["sigma"]
                doa_method._gamma = doa_data.data.get("gamma", None)
                doa_method._threshold = doa_data.data["threshold"]
                doa_method._kernel_type = doa_data.data["kernelType"]
                doa_method._data = doa_data.data["dataPoints"]
            elif doa_data.method == "CITY_BLOCK":
                doa_method = CityBlock()
                doa_method._mean_vector = doa_data.data["meanVector"]
                doa_method._threshold = doa_data.data["threshold"]

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


def predict_onnx(model, preprocessor, dataset: JaqpotpyDataset, request):
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

    if preprocessor:
        onnx_graph = preprocessor
    else:
        onnx_graph = model

    # prepare initial types for preprocessing
    input_feed = {}
    for independent_feature in onnx_graph.graph.input:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
            independent_feature.type.tensor_type.elem_type
        )
        if len(onnx_graph.graph.input) == 1:
            input_feed[independent_feature.name] = dataset.X.values.astype(np_dtype)
        else:
            input_feed[independent_feature.name] = (
                dataset.X[independent_feature.name]
                .values.astype(np_dtype)
                .reshape(-1, 1)
            )
    if preprocessor:
        preprocessor_session = InferenceSession(preprocessor.SerializeToString())
        input_feed = {"input": preprocessor_session.run(None, input_feed)[0]}

    if request.model.doas:
        doas_results = calculate_doas(input_feed, request)
    else:
        doas_results = None

    model_session = InferenceSession(model.SerializeToString())
    for independent_feature in model.graph.input:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
            independent_feature.type.tensor_type.elem_type
        )

    input_feed = {
        model_session.get_inputs()[0].name: input_feed["input"].astype(np_dtype)
    }
    onnx_prediction = model_session.run(None, input_feed)

    if request.model.preprocessors:
        for preprocessor in reversed(request.model.preprocessors):
            preprocessor_name = preprocessor.name
            preprocessor_config = preprocessor.config
            preprocessor_recreated = recreate_preprocessor(
                preprocessor_name, preprocessor_config
            )
            if (
                len(request.model.dependent_features) == 1
                and preprocessor_name != "LabelEncoder"
            ):
                onnx_prediction[0] = preprocessor_recreated.inverse_transform(
                    onnx_prediction[0].reshape(-1, 1)
                )
            onnx_prediction[0] = preprocessor_recreated.inverse_transform(
                onnx_prediction[0]
            )

    if len(request.model.dependent_features) == 1:
        onnx_prediction[0] = onnx_prediction[0].flatten()

    # Probabilities estimation
    probs_list = []
    if request.model.task.lower() in [
        "binary_classification",
        "multiclass_classification",
    ]:
        for instance in onnx_prediction[1]:
            rounded_instance = {k: round(v, 3) for k, v in instance.items()}
            if (
                request.model.preprocessors
                and request.model.preprocessors[0].name == "LabelEncoder"
            ):
                labels = request.model.preprocessors[0].config["classes_"]
                rounded_instance = {labels[k]: v for k, v in rounded_instance.items()}

            probs_list.append(rounded_instance)
    else:
        probs_list = [None for _ in range(len(onnx_prediction[0]))]

    return onnx_prediction[0], probs_list, doas_results
