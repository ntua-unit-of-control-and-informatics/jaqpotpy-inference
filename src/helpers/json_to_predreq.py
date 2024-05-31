def decode(request):
    dataset = request.dataset
    data_entry_all = [item['values']['0'] for item in dataset['dataEntry']]
    return data_entry_all

def decode_with_external(request):
    dataset = request.dataset
    sorted_features = sorted(request.dataset['features'], key=lambda x: int(x['key']))
    names_sorted_by_key = [item['name'] for item in sorted_features]
    input_values = list(dataset['dataEntry'][0]['values'].values())
    input_values = [[i] for i in input_values]
    data_entry_all = dict(zip(names_sorted_by_key, input_values))
    Smiles_input = [data_entry_all.pop('Smiles')][0]
    return Smiles_input, data_entry_all
