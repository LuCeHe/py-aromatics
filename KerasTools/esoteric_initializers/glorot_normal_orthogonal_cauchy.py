import math

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
from tensorflow.python.keras.initializers.initializers_v2 import _compute_fans, _RandomGenerator, _validate_kwargs, \
    _assert_float_dtype, _get_dtype

import tensorflow_probability as tfp

tfd = tfp.distributions


class GlorotOrthogonal(tf.keras.initializers.Orthogonal):

    def __init__(self, gain=1.0, seed=None, mode='fan_in'):
        self.fixed_gain = gain
        self.gain = gain
        self.seed = seed
        self.mode = mode
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=None, **kwargs):
        fan_in, fan_out = _compute_fans(shape)

        scale = self.fixed_gain
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)

        self.gain = math.sqrt(scale)
        super().__call__(**kwargs)

    def get_config(self):
        return {'gain': self.gain, 'seed': self.seed, 'mode': self.mode}


class CauchyOrthogonal(tf.keras.initializers.Initializer):
    """

    Small modification over the Orthogonal initializer with a Cauchy distribution instead of a Gaussian one

    Args:
      gain: multiplicative factor to apply to the orthogonal matrix
      seed: A Python integer. An initializer created with a given seed will
        always produce the same random tensor for a given shape and dtype.

    """

    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized to an orthogonal matrix.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs, support_partition=False)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        # Check the shape
        if len(shape) < 2:
            raise ValueError('The tensor to initialize must be '
                             'at least two-dimensional')
        # Flatten the input shape with the last dimension remaining
        # its original shape so it works for conv2d
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))

        # Generate a random matrix
        # original: a = self._random_generator.random_normal(flat_shape, dtype=dtype)
        dist = tfd.Cauchy(loc=0., scale=1.)
        a = dist.sample(flat_shape)

        # Compute the qr factorization
        q, r = gen_linalg_ops.qr(a, full_matrices=False)
        # Make Q uniform
        d = array_ops.tensor_diag_part(r)
        q *= math_ops.sign(d)
        if num_rows < num_cols:
            q = array_ops.matrix_transpose(q)
        return self.gain * array_ops.reshape(q, shape)

    def get_config(self):
        return {'gain': self.gain, 'seed': self.seed}


class GlorotCauchyOrthogonal(CauchyOrthogonal):

    def __init__(self, gain=1.0, seed=None, mode='fan_in'):
        self.fixed_gain = gain
        self.gain = gain
        self.seed = seed
        self.mode = mode
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=None, **kwargs):
        fan_in, fan_out = _compute_fans(shape)

        scale = self.fixed_gain
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)

        self.gain = math.sqrt(scale)
        super().__call__(**kwargs)

    def get_config(self):
        return {'gain': self.gain, 'seed': self.seed, 'mode': self.mode}
