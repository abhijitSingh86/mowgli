import pickle

import tensorflow as tf
from tensorflow_core.python.keras.models import load_model
import numpy as np

from mowgli.model.create_model import map_to_numpy_array
from mowgli.model.datasets import load_dataset
from mowgli.utils import constants
from mowgli.utils.constants import LABEL_DATA_PATH

VECTORIZER = pickle.load(open(constants.VECTORIZER_PATH, 'rb'))
MODEL = load_model(constants.MODEL_PATH)
LABEL_MAP = map_to_numpy_array(load_dataset(LABEL_DATA_PATH))

def classify(message):
    encoded_matrix = VECTORIZER.transform([message]).toarray()
    result = MODEL.predict(encoded_matrix)
    index = np.argmax(result[0])
    return LABEL_MAP[index, 1], 1.0
