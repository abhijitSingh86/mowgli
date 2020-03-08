import pickle

import tensorflow as tf
import tensorflow_text as text
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras
from tensorflow_core.python.keras import layers

from mowgli.utils import constants


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
    encoded_matrix = vectorizer.fit_transform(dataset)
    return encoded_matrix, vectorizer


def split(data):
    length = int(len(data) * .8)
    return data[0:length], data[length:]


def build_network(bags, labels):
    bags = bags.toarray()
    input_layer = keras.Input(shape=[len(bags[0]), ], name='tokens')
    hidden_layer = layers.Dense(8)(input_layer)
    hidden_layer = layers.Dense(8, activation="softmax")(hidden_layer)
    # print(len(labels[0]))
    output_layer = layers.Dense(2, name='IntentClassification')(hidden_layer)
    print(output_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
                  # Loss function to minimize
                  loss="mean_squared_error",
                  metrics=['mae', 'acc']
                  )
    print('# Fit model on training data')

    train_x, validate_x = split(bags)
    train_y, validate_y = split(labels)
    print('validation sets', validate_x.shape, validate_y.shape)
    print('train sets', train_x.shape, train_y.shape)
    history = model.fit(train_x, train_y,
                        batch_size=2,
                        epochs=3)

    model.save(constants.MODEL_PATH)
    print('\nhistory dict:', history.history)
    return model
