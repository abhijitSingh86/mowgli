import tensorflow as tf
import tensorflow_text as text
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
    vectorizer = CountVectorizer(max_features=vocabulary_count)
    encoded_matrix = vectorizer.fit_transform(dataset)
    return encoded_matrix, vectorizer


def build_network(bags, vectorizer, labels):
    pass
