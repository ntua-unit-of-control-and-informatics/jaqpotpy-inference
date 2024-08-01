from base64 import b64decode
import pickle
import compress_pickle


def decode(raw_model):
    model = b64decode(raw_model)
    try:
        p_mod = compress_pickle.loads(model, compression='bz2')
    except Exception as e:
        p_mod = pickle.loads(model)
    return p_mod
