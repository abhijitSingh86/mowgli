import pickle

import numpy as np
import numpy.testing as npt
import tensorflow as tf
from tensorflow_core.python.keras.models import load_model

from mowgli.model import datasets
from mowgli.model.create_model import map_to_numpy_array
from mowgli.model.datasets import load_dataset
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
    label_arr = map_to_numpy_array(load_dataset('./resources/labels.csv'))
    label_map = dict([[x[1], x[0]] for x in label_arr])
    print(label_map)
    input = np.array([
        ["hi", label_map['greet']],
        ["balu", label_map['greet']],
        ["hola", label_map['greet']],
        ["can i cancel?", label_map['leave_annual_cancel']],
        ["greetings", label_map['greet']],
        ["show me my leave balance", label_map['leave_budget']],
        ["cancel my leaves", label_map['leave_annual_cancel']],
        ["thank you", label_map['thanks']],
        ["stupid you", label_map['insult']],
        ["bye", label_map['goodbye']],
        ["what can you do", label_map['skills']]
    ])

    result_arr = []
    for i in range(0, 5):
        intent_labels = [int(x) for x in input[:, -1].flatten()]
        input_str = input[:, 0:1].flatten()
        print('inputs', input_str, intent_labels)
        vectorizer = pickle.load(open(constants.VECTORIZER_PATH, 'rb'))
        encoded_matrix = vectorizer.transform(input_str).toarray()
        model = load_model(constants.MODEL_PATH)
        result = model.predict(encoded_matrix)
        print('result', np.argmax(result, axis=1))
        result_arr.append(np.sum(np.equal(np.argmax(result, axis=1), np.array(intent_labels))) >= 8)

    assert np.array(result_arr).sum() >= 4
