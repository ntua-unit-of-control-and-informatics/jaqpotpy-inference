class PredictionRequest:

    def __init__(self, dataset, rawModel, additionalInfo = None, doaMatrix = None):
        self.dataset = dataset
        self.rawModel = rawModel
        self.additionalInfo = additionalInfo
        self.doaMatrix = doaMatrix
