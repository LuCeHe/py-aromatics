import numpy as np


def unpackbits(x, num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = 2 ** np.arange(num_bits).reshape([1, num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])


def packbits(b, num_bits):
    rg = np.arange(1, upp_pow + 1)
    return (np.sum((2 ** rg * b) * b, axis=-1) / 2).astype(int)


if __name__ == '__main__':
    vocab_size = 59000
    upp_pow = int(np.log(vocab_size) / np.log(2)) + 1
    print(upp_pow)
    a = np.array([[1, 512, 1000], [0, 58000, 1000]], dtype=np.uint16)

    b = unpackbits(a, upp_pow)

    print(a.shape)
    print(b.shape)
    print(b)

    # and back

    new_a = packbits(b, upp_pow)
    print(new_a)
