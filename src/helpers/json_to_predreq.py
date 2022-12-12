from tornado import httpserver
from tornado import gen
from tornado.ioloop import IOLoop
import tornado.web
from tornado.escape import json_decode, json_encode
from ..entities.prediction_request import PredictionRequest
from ..entities.dataset import Dataset
from ..entities.dataentry import DataEntry
from ..helpers import model_decoder
import numpy as np


def decode(request):
    json_request = json_decode(request.body)
    dataset = json_request['dataset']
    dataEntryAll = [item['values']['0'] for item in dataset['dataEntry']]
    return dataEntryAll
