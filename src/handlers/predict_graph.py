from ..entities.prediction_request import PredictionRequestPydantic
from ..helpers import model_decoder
from rdkit import Chem
import base64
import io
import onnxruntime
import os
import sys
import numpy as np
import torch
current_dir = os.path.dirname(__file__)
software_dir = os.path.abspath(os.path.join(current_dir, '../../../../../JQP/jaqpotpy'))
sys.path.append(software_dir)
from jaqpotpy.descriptors.graph.graph_featurizer import SmilesGraphFeaturizer

def graph_post_handler(request: PredictionRequestPydantic):
    #model = model_decoder.decode(request.model['rawModel'])
    onnx_model = base64.b64decode(request.model['rawModel'])
    ort_session = onnxruntime.InferenceSession(onnx_model)
    feat_config = request.extraConfig['torchConfig']['featurizer']   
    print(feat_config)
    featurizer = SmilesGraphFeaturizer()
    featurizer.load_json_rep(feat_config)
    smiles = request.dataset['input'][0]['SMILES']
    def to_numpy(tensor):
         return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    data = featurizer.featurize(smiles)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data.x),
              ort_session.get_inputs()[1].name: to_numpy(data.edge_index),
              ort_session.get_inputs()[2].name: to_numpy(torch.zeros(data.x.shape[0], dtype=torch.int64))}
    ort_outs = torch.tensor(np.array(ort_session.run(None, ort_inputs)))
    import torch.nn.functional as F
    probs = F.sigmoid(ort_outs)
    preds = (probs > 0.5).int()
    print(preds)