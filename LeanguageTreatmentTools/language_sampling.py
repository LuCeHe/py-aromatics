import tensorflow as tf


def sample_top_k(logits, k):
    assert len(logits.shape) == 3

    _, i_k = tf.math.top_k(logits, k=k)
    i_k = tf.cast(i_k, tf.float32)

    b = tf.shape(logits)[0]
    l = tf.shape(logits)[1]
    p = tf.tile([[1 / k] * k], [b * l, 1])
    mask = tf.random.categorical(tf.math.log(p), 1)

    output_tensor = tf.reshape(mask, (b, l))
    oh = tf.one_hot(indices=output_tensor, depth=k)

    k_sampled = tf.reduce_sum(oh * i_k, axis=2)
    k_sampled = tf.cast(k_sampled, tf.int32)

    return k_sampled
