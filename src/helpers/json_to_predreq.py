def decode(request):
    dataset = request.dataset
    data_entries = dataset['input']
    data_entry_all = [item['values'][0] for item in data_entries]
    return data_entry_all
