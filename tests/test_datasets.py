import pickle

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from tensorflow_core.python.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from mowgli.model import datasets
from mowgli.model.datasets import load_dataset
from mowgli.utils import constants
from mowgli.model.create_model import SENTENCE_MAX_LENGTH


def test_should_load_dataset_with_3_entries():
    actual_labels, actual_features = datasets.load_dataset("tests/resources/dataset.csv")
    npt.assert_array_equal(actual_labels, [2, 1, 0])
    npt.assert_array_equal(actual_features, ['foo bar', 'foobar', 'spaghetti'])


def test_should_tokenize_dataset():
    given_dataset = tf.constant(['foo bar.', 'spaghetti'])

    actual = datasets.tokenize(given_dataset).to_list()
    expected = [[b'foo', b'bar.'], [b'spaghetti']]

    assert expected == actual


def test_should_encode_tokenized_dataset():
    given_dataset = ['foo bar spaghetti', 'spaghetti bar bar']
    actual, tokenizer = datasets.encode_vectorize(given_dataset, 4)
    expected = np.array([[1, 2, 3], [3, 2, 2]])
    npt.assert_array_equal(expected, actual)


def test_model_should_predict_correct_intent():
    labels, sentences = load_dataset('./resources/labels.csv')
    label_map = dict(zip(sentences, labels))
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
        encoded_matrix = vectorizer.texts_to_sequences(input_str)
        encoded_matrix = pad_sequences(encoded_matrix, padding="post", maxlen=SENTENCE_MAX_LENGTH, truncating="post")
        model = load_model(constants.MODEL_PATH)
        result = model.predict(encoded_matrix)
        print('result', np.argmax(result, axis=1),'/n final comparision ',np.sum(np.equal(np.argmax(result, axis=1), np.array(intent_labels))))
        result_arr.append(np.sum(np.equal(np.argmax(result, axis=1), np.array(intent_labels))) >= 7)

    assert np.array(result_arr).sum() >= 4
