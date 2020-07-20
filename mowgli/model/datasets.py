import pickle

import csv
import numpy as np
import tensorflow as tf
import tensorflow_text as text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.keras import layers

from mowgli.utils import constants


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def load_dataset(dataset_path):
    with open(dataset_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        labels = []
        sentences = []
        for line in reader:
            labels.append(line[0])
            sentences.append(",".join(line[1:]))
    return np.array(labels).astype(int), sentences


def tokenize(dataset):
    tokenizer = text.WhitespaceTokenizer()
    return tokenizer.tokenize(dataset)


def persist_vectorizer(vectorizer):
    pickle.dump(vectorizer, open(constants.VECTORIZER_PATH, 'wb'))


def encode_vectorize(dataset, vocabulary_count):
    tokenizer = Tokenizer(num_words=vocabulary_count, oov_token="<OOH_TKN>")
    # encoded_matrix = map(vectorizer.fit_transform, dataset)
    # print('Encoded Matrix ', np.array(list(encoded_matrix)))
    tokenizer.fit_on_texts(dataset)
    return tokenizer.texts_to_sequences(dataset), tokenizer


def split(data):
    length = int(len(data) * .8)
    return data[0:length], data[length:]


def reformat_network_dataset(given_dataset, columne_size):
    label_arr = np.array(given_dataset[:, 0:1], np.int32)
    result_arr = np.zeros([len(given_dataset), columne_size])

    for i, value in enumerate(label_arr):
        result_arr[i][value[0]] = 1

    return given_dataset[:, -1], result_arr


def build_network(train_x, train_y, test_x, test_y, epochs, total, max_length):

    print("total words", total)
    model = tf.keras.Sequential([
        layers.Embedding(total + 1, 64, input_length=max_length),
        layers.Dropout(.1),
        layers.Flatten(),
        layers.Dense(600, activation='relu'),
        layers.Dense(300, activation='relu'),
        layers.Dense(16, activation='softmax')
        ]
    )
    model.compile(optimizer='Adam',  # Optimizer
                  # Loss function to minimize
                  loss="sparse_categorical_crossentropy"
                ,metrics=['acc']
                  )
    model.summary()
    print('# Fit model on training data')
    print('validation sets', test_x.shape, test_y.shape)
    # print('validation sets', test_x, test_y)
    print('train sets', train_x.shape, train_y.shape)
    history = model.fit(train_x, train_y,
                        batch_size=2,
                        epochs=10,
                        validation_data=(test_x, test_y)
                        )
    print('\nhistory dict:', history.history)
    return model
