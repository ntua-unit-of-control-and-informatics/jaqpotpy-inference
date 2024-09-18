from base64 import b64decode
import onnx

def decode(raw_model):
    model = b64decode(raw_model)
    model = onnx.load_from_string(model)
    return model
