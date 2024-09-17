import pandas as pd
from src.helpers.recreate_featurizer import recreate_featurizer
from jaqpotpy.datasets import JaqpotpyDataset


def decode(request):
    df = pd.DataFrame(request.dataset["input"])
    independent_features = request.model["independentFeatures"]
    # smiles_cols = [feature['key'] for feature in independent_features if feature['featureType'] == 'SMILES']
    smiles_cols = [
        feature["key"]
        for feature in independent_features
        if feature["featureType"] == "SMILES"
    ] or None
    x_cols = [
        feature["key"]
        for feature in independent_features
        if feature["featureType"] != "SMILES"
    ]
    featurizers = []
    if request.model["extraConfig"]["featurizers"]:
        for i in range(len(request.model["extraConfig"]["featurizers"])):
            featurizer_name = request.model["extraConfig"]["featurizers"][i]["name"]
            featurizer_config = request.model["extraConfig"]["featurizers"][i]["config"]
            featurizer = recreate_featurizer(featurizer_name, featurizer_config)
            featurizers.append(featurizer)
    else:
        featurizers = None

    dataset = JaqpotpyDataset(
        df=df,
        smiles_cols=smiles_cols,
        x_cols=x_cols,
        task=request.model["task"].lower(),
        featurizer=featurizers[0] if isinstance(featurizers, list) else featurizers,
    )
    return dataset
