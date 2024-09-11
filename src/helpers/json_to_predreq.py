from jaqpotpy.datasets import JaqpotpyDataset
import pandas as pd
from src.helpers.recreate_featurizer import recreate_featurizer

def decode(request):
    df = pd.DataFrame(request.dataset['input'])
    independent_features = request.model['independentFeatures']
    smiles_cols = [feature['key'] for feature in independent_features if feature['featureType'] == 'SMILES']
    x_cols = [feature['key'] for feature in independent_features if feature['featureType'] != 'SMILES']

    if request.model['extraConfig']['featurizers']:
        featurizer_name = request.model['extraConfig']['featurizers'][0]['name']
        featurizer_config = request.model['extraConfig']['featurizers'][0]['config']
        featurizer = recreate_featurizer(featurizer_name, featurizer_config)

    dataset = JaqpotpyDataset(df=df, smiles_cols=smiles_cols,  x_cols=x_cols,
                              task=request.model['task'].lower(), featurizer=featurizer)
    return dataset
