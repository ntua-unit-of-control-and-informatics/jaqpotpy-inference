def decode(request):
    dataset = request.dataset
    dataEntryAll = [item['values']['0'] for item in dataset['dataEntry']]
    return dataEntryAll
