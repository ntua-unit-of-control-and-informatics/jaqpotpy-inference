import pandas as pd
from src.helpers.recreate_featurizer import recreate_featurizer
from jaqpotpy.datasets import JaqpotpyDataset


def decode(request):
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
        for i in range(len(request.model.featurizers)):
            featurizer_name = request.model.featurizers[i]["name"]
            featurizer_config = request.model.featurizers[i]["config"]
            featurizer = recreate_featurizer(featurizer_name, featurizer_config)
            featurizers.append(featurizer)
    else:
        featurizers = None

    dataset = JaqpotpyDataset(
        df=df,
        smiles_cols=smiles_cols,
        x_cols=x_cols,
        task=request.model.task.lower(),
        featurizer=featurizers,
    )
    if len(request.model.selected_features) > 0:
        dataset.select_features(SelectionList=request.model.selected_features)
    return dataset, jaqpot_row_ids
