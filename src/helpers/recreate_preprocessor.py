import numpy as np
from jaqpotpy.transformers import LogTransformer


def recreate_preprocessor(preprocessor_name, preprocessor_config):
    if preprocessor_name == "LogTransformer":
        preprocessor = LogTransformer()
    else:
        preprocessor_class = getattr(
            __import__("sklearn.preprocessing", fromlist=[preprocessor_name]),
            preprocessor_name,
        )
        preprocessor = preprocessor_class()
        for attr, value in preprocessor_config.items():
            if attr != "class":  # skip the class attribute
                if isinstance(value, list):
                    value = np.array(value)
                setattr(preprocessor, attr, value)
    return preprocessor
