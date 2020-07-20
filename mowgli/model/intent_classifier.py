import pickle

import tensorflow as tf
from tensorflow_core.python.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from mowgli.model.datasets import load_dataset
from mowgli.utils import constants
from mowgli.utils.constants import LABEL_DATA_PATH
from mowgli.model.create_model import SENTENCE_MAX_LENGTH


VECTORIZER = pickle.load(open(constants.VECTORIZER_PATH, 'rb'))
MODEL = load_model(constants.MODEL_PATH)
LABELS, LABEL_SENTENCES = load_dataset(LABEL_DATA_PATH)


def classify(message):

    encoded_matrix = VECTORIZER.texts_to_sequences([message])
    encoded_matrix = pad_sequences(encoded_matrix, padding="post", maxlen=SENTENCE_MAX_LENGTH, truncating="post")
    result = MODEL.predict(encoded_matrix)
    index = np.argmax(result[0])
    return LABEL_SENTENCES[index], 1.0
