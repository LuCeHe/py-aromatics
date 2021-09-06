import numpy as np


def unpad_sequence(padded_sequence, padding='pre', value=0.):
    """ removes the extra zeros up to the longest non zero sequence in the array """
    original_shape = padded_sequence.shape
    x = padded_sequence
    x = x if len(original_shape) == 2 else np.reshape(x, (-1, original_shape[-1]))

    x = x if padding == 'post' else x[:, ::-1]
    trimmable_zeros = np.sum(1 - np.prod(x == value, axis=0))
    unpadded = x[:, :trimmable_zeros]

    unpadded = unpadded if padding == 'post' else unpadded[:, ::-1]
    unpadded = unpadded if len(original_shape) == 2 \
        else np.reshape(unpadded, (*original_shape[:-1], unpadded.shape[-1]))
    return unpadded


if __name__ == '__main__':
    from tensorflow.python.keras.preprocessing.sequence import pad_sequences

    a = [[1, 2, 2], [3, 3, ]]

    print('\n  Unpadding pre-padded sequences')
    x = pad_sequences(a, maxlen=5, padding='pre')
    unpadded = unpad_sequence(x, padding='pre')
    print(x)
    print(unpadded)

    print('\n  Unpadding post-padded sequences')
    x = pad_sequences(a, maxlen=5, padding='post')
    unpadded = unpad_sequence(x, padding='post')
    print(x)
    print(unpadded)

    print('\n  Unpadding axis==2')

    a = [[1, 2, 2], [3, 3, ], [1, 1, 2], [2, 4, ]]
    x = pad_sequences(a, maxlen=5, padding='post')
    x = np.reshape(x, [2, 2, 5])
    unpadded = unpad_sequence(x, padding='post')
    print(x)
    print(unpadded)
