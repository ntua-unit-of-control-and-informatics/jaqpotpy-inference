import numpy as np


def recreate_featurizer(featurizer_name, featurizer_config):
    featurizer_class = getattr(
        __import__("jaqpotpy.descriptors.molecular", fromlist=[featurizer_name]),
        featurizer_name,
    )
    featurizer = featurizer_class()
    for attr, value in featurizer_config.items():
        if attr != "class":  # skip the class attribute
            setattr(featurizer, attr, value)
    return featurizer
