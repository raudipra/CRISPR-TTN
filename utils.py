import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

def euclidean_distance(x, y):
    dist = tf.square(x - y)
    dist = tf.reduce_sum(dist, 1, keepdims=True)
    dist = tf.sqrt(tf.dtypes.cast(dist, tf.float32))
    return dist

def manhattan_distance(x, y):
    dist = tf.abs(x - y)
    dist = tf.reduce_sum(dist, 1, keepdims=True)
    return dist

@tf.function
def one_hot(sequence):
    # Why N? for one encoding purpose, last character = [0, 0, 0, 0]
    VOCAB = ["A", "G", "T", "N"]
    string_lookup = StringLookup(vocabulary=VOCAB)

    sequence = string_lookup(sequence)
    C = tf.constant(5)
    one_hot_matrix = tf.one_hot(
        sequence,
        C,
        on_value=1.0,
        off_value=0.0,
        axis =-1
    )
    return one_hot_matrix[:, 1:]

def convert_embed_to_label(y, margin=1.0):
    dist = euclidean_distance(y[:, 0], y[:, 1])
    new_y = dist <= margin
    new_y = tf.squeeze(tf.cast(new_y, tf.int32), axis=1)
    return new_y