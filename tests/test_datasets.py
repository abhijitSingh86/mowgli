import numpy as np
import tensorflow_text as text
import tensorflow as tf
import numpy.testing as npt

from mowgli.model import datasets


def test_should_load_dataset_with_3_entries():
    actual_dataset = datasets.load_dataset("tests/resources/dataset.csv")
    actual_labels, actual_features = next(iter(actual_dataset.batch(3)))

    npt.assert_array_equal(actual_labels, np.array([2, 1, 0], dtype=int))
    npt.assert_array_equal(actual_features, np.array([b'foo bar', b'foobar', b'spaghetti'], dtype=object))


def test_should_tokenize_dataeset():
    given_dataset = tf.constant(['foo bar.', 'spaghetti'])

    actual = datasets.tokenize(given_dataset).to_list()
    expected = [[b'foo', b'bar.'], [b'spaghetti']]

    assert expected == actual
