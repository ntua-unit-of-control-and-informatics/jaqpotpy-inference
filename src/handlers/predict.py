import tornado.web
from tornado.escape import json_decode, json_encode
from ..entities.prediction_request import PredictionRequest
from ..helpers import model_decoder, json_to_predreq, doa_calc


class ModelHandler(tornado.web.RequestHandler):
    # @asynchronous
    # @gen.engine
    # @gen.coroutine
    def post(self):
        json_request = json_decode(self.request.body)
        pred_request = PredictionRequest(json_request['dataset'], json_request['rawModel'])
        rawModel = pred_request.rawModel[0]
        model = model_decoder.decode(rawModel)
        dataEntryAll = json_to_predreq.decode(self.request)

        _ = model(dataEntryAll)

        if isinstance(model.prediction[0], list):
            results = {model.Y[i]: [item[i] for item in model.prediction] for i in range(len(model.prediction[0]))}
        elif isinstance(model.prediction, list):
            if isinstance(model.Y, list):
                results = {model.Y[0]: [item for item in model.prediction]}
            else:
                results = {model.Y: [item for item in model.prediction]}
        else:
            results = {model.Y: [item for item in model.prediction]}


        if model.doa:
            results['AD'] = model.doa.IN
        else:
            results['AD'] = [None for _ in range(len(model.prediction))]

        if model.probability:
            results['Probabilities'] = [list(prob) for prob in model.probability]
        else:
            results['Probabilities'] = [[] for _ in range(len(model.prediction))]

        finalAll = {"predictions": [dict(zip(results, t)) for t in zip(*results.values())]}

        self.write(json_encode(finalAll))
