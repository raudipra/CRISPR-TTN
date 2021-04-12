import tensorflow as tf


def get_most_frequent_label(samples):
    samples = tf.squeeze(samples, axis=1)
    unique, _, count = tf.unique_with_counts(samples)
    max_idx = tf.math.argmax(count)
    
    return unique[max_idx], count[max_idx]
    
def euclidean_distance(x, y):
    dist = tf.square(x - y)
    dist = tf.reduce_sum(dist, 1, keepdims=True)
    dist = tf.sqrt(tf.dtypes.cast(dist, tf.float32))
    return dist

def manhattan_distance(x, y):
    dist = tf.abs(x - y)
    dist = tf.reduce_sum(dist, 1, keepdims=True)
    return dist
