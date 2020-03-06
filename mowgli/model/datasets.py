import tensorflow as tf
import tensorflow_text as text
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras
from tensorflow.keras import layers


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


def split(data):
    length = int(len(data)*.8)
    return data[0:length], data[length:]

def build_network(bags, vectorizer, labels):
    bags = bags.toarray()
    input = keras.Input(shape=[len(bags[0]), ], name='tokens')
    print(input)

    x = layers.Dense(8)(input)
    x = layers.Dense(8, activation="softmax")(x)
    outputs = layers.Dense(len(labels), name='IntentClassification')(x)
    print(outputs)
    model = keras.Model(inputs=input, outputs=outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
                  # Loss function to minimize
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  # List of metrics to monitor
                  metrics=['sparse_categorical_accuracy'])
    print('# Fit model on training data')

    train_x,validate_x = split(bags)
    train_y,validate_y = split(labels)
    history = model.fit(train_x, train_y,
                        batch_size=2,
                        epochs=3,
                        # We pass some validation for
                        # monitoring validation loss and metrics
                        # at the end of each epoch
                        validation_data=(validate_x, validate_y))

    print('\nhistory dict:', history.history)
    return model
