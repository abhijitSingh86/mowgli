import pickle

import numpy as np
import numpy.testing as npt
import tensorflow as tf
from tensorflow_core.python.keras.models import load_model

from mowgli.model import datasets
from mowgli.model.datasets import reformat_network_dataset, split
from mowgli.utils import constants


def test_should_load_dataset_with_3_entries():
    actual_dataset = datasets.load_dataset("tests/resources/dataset.csv")
    actual_labels, actual_features = next(iter(actual_dataset.batch(3)))

    npt.assert_array_equal(actual_labels, np.array([2, 1, 0], dtype=int))
    npt.assert_array_equal(actual_features, np.array([b'foo bar', b'foobar', b'spaghetti'], dtype=object))


def test_should_tokenize_dataset():
    given_dataset = tf.constant(['foo bar.', 'spaghetti'])

    actual = datasets.tokenize(given_dataset).to_list()
    expected = [[b'foo', b'bar.'], [b'spaghetti']]

    assert expected == actual


def test_should_encode_tokenized_dataset():
    given_dataset = ['foo bar spaghetti', 'spaghetti bar bar']
    actual, vectorizer = datasets.encode_vectorize(given_dataset, 3)
    expected = np.array([[1, 1, 1], [2, 0, 1]])
    npt.assert_array_equal(expected, actual.toarray())


def test_model_should_predict_correct_intent():
    input_str = ["hi, balu", "hola", "greetings", "show me my leave balance", "cancel my leaves", "thank you", "stupid you", "bye", "what can you do"]
    intent_labels = [2, 2, 2, 4, 4, 3, 8, 10, 11]
    vectorizer = pickle.load(open(constants.VECTORIZER_PATH, 'rb'))
    encoded_matrix = vectorizer.transform(input_str).toarray()
    model = load_model(constants.MODEL_PATH)
    print('Encoded Matrix', encoded_matrix)
    result = model.predict(encoded_matrix)
    print('result', result)
    print('result', np.argmax(result, axis=1))
    assert np.sum(np.equal(np.argmax(result, axis=1), np.array(intent_labels))) >= 8


test_model_should_predict_correct_intent()
