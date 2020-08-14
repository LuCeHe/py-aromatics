import tensorflow as tf


def dynamic_zeros(x, d):
    batch_size = tf.shape(x)[0]
    return tf.zeros(tf.stack([batch_size, 1, d]))


def dynamic_ones(x, d):
    batch_size = tf.shape(x)[0]
    return tf.ones(tf.stack([batch_size, 1, d]))


def dynamic_fill(x, d, value):
    batch_size = tf.shape(x)[0]
    return tf.fill(tf.stack([batch_size, 1, d]), value)


def dynamic_filler(batch_as, d, value):
    batch_size = tf.shape(batch_as)[0]
    return tf.fill(tf.stack([batch_size, d]), value)


def dynamic_one_hot(x, d, pos):
    batch_size = tf.shape(x)[0]
    one_hots = tf.ones(tf.stack([batch_size, 1, d])) * tf.one_hot(pos, d)
    return one_hots


def slice_(x):
    return x[:, :-1, :]


def slice_from_to(x, initial, final):
    # None can be used where initial or final, so
    # [1:] = [1:None]
    return x[:, initial:final, ...]


def clip_layer(inputs, min_value, max_value):
    eps = .5e-6
    clipped_point = tf.clip_by_value(inputs, min_value + eps, max_value - eps)
    return clipped_point


def replace_column(matrix, new_column, r):
    dynamic_index = tf.cast(tf.squeeze(r), dtype=tf.int64)
    matrix = tf.cast(matrix, dtype=tf.float32)
    new_column = tf.cast(new_column, dtype=tf.float32)
    num_cols = tf.shape(matrix)[1]
    # new_matrix = tf.assign(matrix[:, dynamic_index], new_column)
    index_row = tf.stack([tf.eye(num_cols, dtype=tf.float32)[dynamic_index, :]])
    old_column = matrix[:, dynamic_index]
    new = tf.matmul(tf.stack([new_column], axis=1), index_row)
    old = tf.matmul(tf.stack([old_column], axis=1), index_row)
    new_matrix = (matrix - old) + new
    return new_matrix


def log_base_n(x, n=2):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(n, dtype=numerator.dtype))
    return numerator / denominator
