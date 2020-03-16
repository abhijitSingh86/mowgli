import pickle

import numpy as np
import tensorflow as tf
import tensorflow_text as text
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow_core.python import keras
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


def parse(line):
    split_line = tf.strings.split(line, sep=',', maxsplit=1)
    return tf.strings.to_number(split_line[0], out_type=tf.dtypes.int32), split_line[1]


def load_dataset(dataset_path):
    return tf.data.TextLineDataset(dataset_path).map(parse)


def tokenize(dataset):
    tokenizer = text.WhitespaceTokenizer()
    return tokenizer.tokenize(dataset)


def persist_vectorizer(vectorizer):
    pickle.dump(vectorizer, open(constants.VECTORIZER_PATH, 'wb'))


def encode_vectorize(dataset, vocabulary_count):
    vectorizer = CountVectorizer(max_features=vocabulary_count)
    # encoded_matrix = map(vectorizer.fit_transform, dataset)
    # print('Encoded Matrix ', np.array(list(encoded_matrix)))
    encoded_matrix = vectorizer.fit_transform(dataset)
    return encoded_matrix, vectorizer


def split(data):
    length = int(len(data) * .8)
    return data[0:length], data[length:]


def reformat_network_dataset(given_dataset, columne_size):
    label_arr = np.array(given_dataset[:, 0:1], np.int32)
    result_arr = np.zeros([len(given_dataset), columne_size])

    for i, value in enumerate(label_arr):
        result_arr[i][value[0]] = 1

    return given_dataset[:, -1], result_arr


def build_network(train_x, train_y, test_x, test_y, epochs):
    input_layer = keras.Input(shape=[train_x.shape[1], ], name='tokens')
    hidden_layer = layers.Dense(150, activation="relu")(input_layer)
    hidden_layer = layers.Dense(70, activation="relu")(hidden_layer)
    # hidden_layer = layers.Dense(20, activation="relu")(hidden_layer)
    # print(len(labels[0]))
    output_layer = layers.Dense(len(train_y[0]), name='IntentClassification',
                                activation="softmax")(hidden_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='Adam',  # Optimizer
                  # Loss function to minimize
                  loss="categorical_crossentropy",
                  metrics=['mae', 'acc']  # ,recall_m,precision_m, f1_m]
                  )
    print('# Fit model on training data')
    print('validation sets', test_x.shape, test_y.shape)
    # print('validation sets', test_x, test_y)
    print('train sets', train_x.shape, train_y.shape)
    history = model.fit(train_x.toarray(), train_y,
                        batch_size=2,
                        epochs=epochs,
                        validation_data=(test_x.toarray(), test_y)
                        )
    print('\nhistory dict:', history.history)
    return model
