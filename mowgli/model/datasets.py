import tensorflow as tf
import tensorflow_text as text
import numpy as np
import tensorflow_datasets as tfds
from sklearn.feature_extraction.text import CountVectorizer


def parse(line):
    split_line = tf.strings.split(line, sep=',', maxsplit=1)
    return tf.strings.to_number(split_line[0], out_type=tf.dtypes.int32), split_line[1]


def load_dataset(dataset_path):
    return tf.data.TextLineDataset(dataset_path).map(parse)


def tokenize(dataset):
    tokenizer = text.WhitespaceTokenizer()
    return tokenizer.tokenize(dataset)


def encode_vectorize(dataset, vocabulary_count):
    # map over all elements in dataset
    # each element of dataset needs to go through count vectorizer
    # dataset.map(vecatorizer.aksdjalkjsda)
    vectorizer = CountVectorizer(max_features=vocabulary_count)
    encodedMatrix = vectorizer.fit_transform(dataset)
    return encodedMatrix, vectorizer
