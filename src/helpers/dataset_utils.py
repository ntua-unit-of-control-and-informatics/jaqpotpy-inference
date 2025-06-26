import pandas as pd
from jaqpot_api_client import PredictionRequest
from jaqpotpy.datasets.jaqpot_tensor_dataset import JaqpotTensorDataset

from src.helpers.recreate_featurizer import recreate_featurizer
from jaqpotpy.datasets import JaqpotTabularDataset


def build_tabular_dataset_from_request(request: PredictionRequest):
    df = pd.DataFrame(request.dataset.input)
    jaqpot_row_ids = []
    for i in range(len(df)):
        jaqpot_row_ids.append(df.iloc[i]["jaqpotRowId"])
    independent_features = request.model.independent_features
    smiles_cols = [
        feature.key
        for feature in independent_features
        if feature.feature_type == "SMILES"
    ] or None
    x_cols = [
        feature.key
        for feature in independent_features
        if feature.feature_type != "SMILES"
    ]
    featurizers = []
    if request.model.featurizers:
        for featurizer in request.model.featurizers:
            featurizer_name = featurizer.name
            featurizer_config = featurizer.config
            featurizer = recreate_featurizer(featurizer_name, featurizer_config)
            featurizers.append(featurizer)
    else:
        featurizers = None

    dataset = JaqpotTabularDataset(
        df=df,
        smiles_cols=smiles_cols,
        x_cols=x_cols,
        task=request.model.task,
        featurizer=featurizers,
    )
    if len(request.model.selected_features) > 0:
        dataset.select_features(SelectColumns=request.model.selected_features)
    return dataset, jaqpot_row_ids


def build_tensor_dataset_from_request(request: PredictionRequest):
    df = pd.DataFrame(request.dataset.input)
    jaqpot_row_ids = []
    for i in range(len(df)):
        jaqpot_row_ids.append(df.iloc[i]["jaqpotRowId"])
    independent_features = request.model.independent_features
    x_cols = [feature.key for feature in independent_features]

    dataset = JaqpotTensorDataset(
        df=df,
        x_cols=x_cols,
        task=request.model.task,
    )
    return dataset, jaqpot_row_ids
