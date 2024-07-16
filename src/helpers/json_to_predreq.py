def decode(request, model):
    df = pd.DataFrame(request.dataset['input'])
    independentFeatures = request.model['independentFeatures']
    smiles_cols = [feature['key'] for feature in independentFeatures if feature['featureType'] == 'SMILES']
    x_cols = [feature['key'] for feature in independentFeatures if feature['featureType'] != 'SMILES']
    dataset = JaqpotpyDataset(df=df, smiles_cols=smiles_cols,  x_cols=x_cols,
                              task=model.task, featurizer=model.featurizer)
    return dataset