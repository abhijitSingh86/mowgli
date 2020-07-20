import numpy as np

from mowgli.model import datasets
from mowgli.model.datasets import load_dataset, build_network
from mowgli.utils import constants
from mowgli.utils.constants import TRAIN_DATA_PATH, TEST_DATA_PATH, LABEL_DATA_PATH
from tensorflow.keras.preprocessing.sequence import pad_sequences

SENTENCE_MAX_LENGTH = 15


def reshape_Labels(labels):
    return np.reshape(np.array(labels), (len(labels), 1))


def create_model():
    train_labels, train_sentences = load_dataset(TRAIN_DATA_PATH)
    test_labels, test_sentences = load_dataset(TEST_DATA_PATH)
    label_data = load_dataset(LABEL_DATA_PATH)
    encoded, tokenizer = datasets.encode_vectorize(train_sentences, 10000)
    total_words = len(tokenizer.word_index)
    encoded = pad_sequences(encoded, padding="post", maxlen=SENTENCE_MAX_LENGTH, truncating="post")
    test_encoded = pad_sequences(tokenizer.texts_to_sequences(test_sentences), padding="post",
                                 maxlen=SENTENCE_MAX_LENGTH, truncating="post")
    datasets.persist_vectorizer(tokenizer)
    model = build_network(encoded, reshape_Labels(train_labels), test_encoded,
                          reshape_Labels(test_labels), epochs=5, total=total_words, max_length=SENTENCE_MAX_LENGTH)
    model.save(constants.MODEL_PATH)


if __name__ == '__main__':
    create_model()