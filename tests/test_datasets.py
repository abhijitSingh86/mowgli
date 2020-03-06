import numpy as np
import numpy.testing as npt
import tensorflow as tf

from mowgli.model import datasets


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


def test_model_should_return_correct_intent():
    given_dataset = ['hey balu', 'hello balu', 'hi balu', 'good afternoon', 'good day', 'good evening', 'good morning',
                     'moin', 'ohai', 'Hi', 'Hey', 'Hi bot']
    actual, vectorizer = datasets.encode_vectorize(given_dataset, len(given_dataset))
    model = datasets.build_network(actual, vectorizer, np.ones(len(given_dataset)))

    input_text = "hello"
    output = 1
