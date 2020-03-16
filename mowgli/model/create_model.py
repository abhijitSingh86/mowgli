import numpy as np

from mowgli.model import datasets
from mowgli.model.datasets import load_dataset, reformat_network_dataset, build_network
from mowgli.utils import constants
from mowgli.utils.constants import TRAIN_DATA_PATH, TEST_DATA_PATH, LABEL_DATA_PATH


def map_to_numpy_array(dataset):
    arr = []
    for data in iter(dataset):
        arr.append([data[0].numpy(), data[1].numpy().decode('utf-8')])
    return np.array(arr)


def create_model():
    train_data = map_to_numpy_array(load_dataset(TRAIN_DATA_PATH))
    test_data = map_to_numpy_array(load_dataset(TEST_DATA_PATH))
    label_data = np.array(map_to_numpy_array(load_dataset(LABEL_DATA_PATH))[:, 0:1], np.int32)
    print(label_data)
    labels_count = np.max(label_data) + 1
    message_train_x, train_y = reformat_network_dataset(train_data, labels_count)
    message_test_x, test_y = reformat_network_dataset(test_data, labels_count)
    concat = np.concatenate([message_test_x, message_train_x])
    encoded, vectorizer = datasets.encode_vectorize(concat, 100000)
    datasets.persist_vectorizer(vectorizer)
    model = build_network(encoded[len(message_test_x):], train_y, encoded[0:len(message_test_x)],
                          test_y, epochs=5)
    model.save(constants.MODEL_PATH)

# UnComment Me when running locally for model creation
# create_model()
