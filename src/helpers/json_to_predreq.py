import numpy as np
import pandas as pd

def decode_smiles(request):
    """
    Decode SMILES from the given request.

    Args:
        request (dict): The request object containing the dataset.

    Returns:
        list: A list of decoded SMILES strings.
    """
    dataset = request.dataset
    data_entry_all = [item['values']['0'] for item in dataset['dataEntry']]
    return data_entry_all

def decode_smiles_external(request):
    """
    Decode SMILES from the given request and return the SMILES input and the remaining data entry.

    Args:
        request (dict): The request containing the dataset.

    Returns:
        tuple: A tuple containing the SMILES input and the remaining data entry.
    """
    dataset = request.dataset
    sorted_features = sorted(request.dataset['features'], key=lambda x: int(x['key']))
    names_sorted_by_key = [item['name'] for item in sorted_features]
    input_values = list(dataset['dataEntry'][data_instance]['values'].values() for data_instance in range(len(dataset['dataEntry'])))
    zipped_data = list(zip(*input_values))
    data_entry_all = dict(zip(names_sorted_by_key, zipped_data)) 
    Smiles_input = [data_entry_all.pop('Smiles')][0]
    return Smiles_input, data_entry_all

def decode_only_external(request):
    """
    Decodes the request object and returns a DataFrame containing the values of the requested features.

    Args:
        request (dict): The request object containing the dataset and additional information.

    Returns:
        pandas.DataFrame: A DataFrame containing the values of the requested features.

    """
    dataset = request.dataset
    expected_features_order = request.additionalInfo['fromUser']['inputSeries']
    features_names_by_key = sorted(request.dataset['features'], key=lambda d: d['key'])
    features_names = list(feature['name'] for feature in features_names_by_key) 
    features_values = list(dataset['dataEntry'][data_instance]['values'].values() for data_instance in range(len(dataset['dataEntry'])))
    zipped_data = list(zip(*features_values))
    input_dict = dict(zip(features_names, zipped_data))
    data_entry_all = np.array(list(input_dict.values())).T    
    data_entry_all_df = pd.DataFrame(data_entry_all, columns=features_names)[expected_features_order]
    return data_entry_all_df