from base64 import b64decode
import pickle
import compress_pickle

def decode(rawModel):
    model = b64decode(rawModel)
    try:
        p_mod = compress_pickle.loads(model, compression='bz2')
    except Exception as e:
        p_mod = pickle.loads(model)
    return p_mod
