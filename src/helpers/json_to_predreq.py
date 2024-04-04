def decode(request):
    dataset = request.dataset
    data_entry_all = [item['values']['0'] for item in dataset['dataEntry']]
    return data_entry_all
