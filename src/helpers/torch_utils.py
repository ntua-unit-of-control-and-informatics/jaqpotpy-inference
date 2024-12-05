import torch.nn.functional as f
from jaqpotpy.api.openapi.models.model_task import ModelTask


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def torch_binary_classification(target_name, output, inp):
    proba = f.sigmoid(output).squeeze().tolist()
    pred = int(proba > 0.5)
    # UI Results
    results = {
        "jaqpotMetadata": {
            "probabilities": [round((1 - proba), 3), round(proba, 3)],
            "jaqpotRowId": inp["jaqpotRowId"],
        }
    }
    if "jaqpotRowLabel" in inp:
        results["jaqpotMetadata"]["jaqpotRowLabel"] = inp["jaqpotRowLabel"]
    results[target_name] = pred
    return results


def torch_regression(target_name, output, inp):
    pred = [output.squeeze().tolist()]
    results = {"jaqpotMetadata": {"jaqpotRowId": inp["jaqpotRowId"]}}
    if "jaqpotRowLabel" in inp:
        results["jaqpotMetadata"]["jaqpotRowLabel"] = inp["jaqpotRowLabel"]
    results[target_name] = pred
    return results


def generate_prediction_response(model_task, target_name, out, row_id):
    if model_task == ModelTask.BINARY_CLASSIFICATION:
        return torch_binary_classification(target_name, out, row_id)
    elif model_task == ModelTask.REGRESSION:
        return torch_regression(target_name, out, row_id)
    else:
        raise ValueError(
            "Only BINARY_CLASSIFICATION and REGRESSION tasks are supported"
        )
