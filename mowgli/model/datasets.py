import tensorflow as tf


def parse(line):
    split_line = tf.strings.split(line, sep=',', maxsplit=1)
    return tf.strings.to_number(split_line[0], out_type=tf.dtypes.int32), split_line[1]


def load_dataset(_dataset_path):
    return tf.data.TextLineDataset(_dataset_path).map(parse)
