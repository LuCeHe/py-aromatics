import tensorflow as tf
import numpy as np


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


# https://stackoverflow.com/questions/39875674/is-there-a-built-in-function-in-tensorflow-for-shuffling-or-permutating-tensors
def tf_shuffle_axis(value, axis=0, seed=None, name=None):
    perm = list(range(len(value.shape)))
    tf_perm = tf.range(tf.rank(value))
    perm[axis], perm[0] = perm[0], perm[axis]
    new_perm = tf.gather(tf_perm, perm)
    # new_perm = tf.concat([perm[0], perm[axis]], axis=0)
    batch = tf.transpose(value, perm=new_perm)
    value = tf.gather(batch, tf.random.shuffle(tf.range(tf.shape(batch)[0])))
    value = tf.transpose(value, perm=perm)
    return value


def snake(logits, frequency=1):
    """Snake activation to learn periodic functions.
    https://arxiv.org/abs/2006.08195

    original code:
    https://www.tensorflow.org/addons/api_docs/python/tfa/activations/snake
    Args:
        logits: Input tensor.
        frequency: A scalar, frequency of the periodic part.
    Returns:
        Tensor of the same type and shape as `logits`.
    """

    logits = tf.convert_to_tensor(logits)
    frequency = tf.cast(frequency, logits.dtype)

    return logits + (1 - tf.cos(2 * frequency * logits)) / (2 * frequency)


def sample_axis(tensor, max_dim=1024, return_deshuffling=False, axis=1):
    if tensor.shape[axis] > max_dim:
        newdim_inp = sorted(np.random.choice(tensor.shape[axis], max_dim, replace=False))
        out_tensor = tf.gather(tensor, indices=newdim_inp, axis=axis)
    else:
        out_tensor = tensor

    if not return_deshuffling:
        return out_tensor

    else:
        if tensor.shape[axis] > max_dim:
            remaining_indices = list(set(range(tensor.shape[axis])).difference(set(newdim_inp)))

            shuffled_indices = newdim_inp + remaining_indices
            deshuffle_indices = np.array(shuffled_indices).argsort()

            remainder = tf.gather(tensor, indices=remaining_indices, axis=axis)
        else:
            remainder, deshuffle_indices = None, None

        return out_tensor, remainder, deshuffle_indices


def desample_axis(sample, remainder, deshuffle_indices, axis = 1):
    if not remainder is None:
        concat = tf.concat([sample, remainder], axis=axis)
        deshuffled = tf.gather(concat, indices=deshuffle_indices, axis=axis)
    else:
        deshuffled = sample

    return deshuffled


def test_shuffling():
    t = tf.random.uniform((2, 3, 4)).numpy()
    st = tf_shuffle_axis(t, axis=2)
    print(t)
    print(st)


def test_sampling_desampling():

    test_several_samples = False
    test_choosing_axis = False
    test_deslice = True

    if test_several_samples:
        print('-' * 20)
        t = tf.random.uniform((2, 34))
        st, remainder, deshuffle_indices = sample_axis(t, max_dim=4, return_deshuffling=True)
        print('original shape:', t.shape)
        print('sample shape:  ', st.shape)
        print('reminder shape:', remainder.shape)
        print(deshuffle_indices)
        dst = desample_axis(st, remainder, deshuffle_indices)
        print('Is the desampled tensor equal to how it was at the beginning?', np.all(dst == t))

    if test_choosing_axis:
        for axis in [0, 1, 2]:
            print('-' * 20)

            t = tf.random.uniform((2, 3, 4))
            st, remainder, deshuffle_indices = sample_axis(t, max_dim=1, return_deshuffling=True, axis=axis)
            print('original shape:', t.shape)
            print('sample shape:  ', st.shape)
            print('reminder shape:', remainder.shape)
            print(deshuffle_indices)
            dst = desample_axis(st, remainder, deshuffle_indices, axis=axis)
            print('desampld shape:', dst.shape)
            print('Is the desampled tensor equal to how it was at the beginning?', np.all(dst==t))

    if test_deslice:
        print('-' * 20)

        deslice_axis=[1,2]
        t = tf.random.uniform((2, 3, 4, 5))
        st = t
        reminders = []
        deshuffles = []
        for axis in deslice_axis:
            st, remainder, deshuffle_indices = sample_axis(st, max_dim=1, return_deshuffling=True, axis=axis)
            reminders.append(remainder)
            deshuffles.append(deshuffle_indices)

            print('original shape:', t.shape)
            print('sample shape:  ', st.shape)
            print('reminder shape:', remainder.shape)
            print(deshuffle_indices)

        for j, _ in enumerate(deslice_axis):
            i = -j - 1
            st = desample_axis(st, reminders[i], deshuffles[i], axis=deslice_axis[i])
            print('desampld shape:', st.shape)
        print('Is the desampled tensor equal to how it was at the beginning?', np.all(st==t))

if __name__ == '__main__':
    test_sampling_desampling()
