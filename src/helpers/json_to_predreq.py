def decode(request):
    dataset = request.dataset
    model = request.model

    keys = [feature['key'] for feature in model['independentFeatures']]
    transformed_values = [[data[key] for key in keys] for data in dataset['input']]

    # TODO fix to support multiple rows
    return transformed_values[0]
