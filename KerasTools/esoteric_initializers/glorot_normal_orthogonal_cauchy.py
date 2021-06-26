import math

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
from tensorflow.python.keras.initializers.initializers_v2 import _compute_fans, _RandomGenerator, _validate_kwargs, \
    _assert_float_dtype, _get_dtype, VarianceScaling

import tensorflow_probability as tfp

tfd = tfp.distributions
_PARTITION_SHAPE = 'partition_shape'


class GlorotOrthogonal(tf.keras.initializers.Orthogonal):

    def __init__(self, gain=1.0, seed=None, mode='fan_avg'):
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
        return super().__call__(shape, dtype, **kwargs)

    def get_config(self):
        return {'gain': self.gain, 'seed': self.seed, 'mode': self.mode}


def orthogonalize(initial_initializer):
    shape = initial_initializer.shape

    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))

    a = array_ops.reshape(initial_initializer, flat_shape)
    # Compute the qr factorization
    q, r = gen_linalg_ops.qr(a, full_matrices=False)
    # Make Q uniform
    d = array_ops.tensor_diag_part(r)
    q *= math_ops.sign(d)
    if num_rows < num_cols:
        q = array_ops.matrix_transpose(q)
    orthogonal_initializer = array_ops.reshape(q, shape)
    return orthogonal_initializer


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
        # Generate a random matrix
        # original: a = self._random_generator.random_normal(flat_shape, dtype=dtype)
        dist = tfd.Cauchy(loc=0., scale=1.)
        initial_initializer = dist.sample(shape) / 5

        orhogonal_cauchy = orthogonalize(initial_initializer)
        return self.gain * orhogonal_cauchy

    def get_config(self):
        return {'gain': self.gain, 'seed': self.seed}


class GlorotCauchyOrthogonal(CauchyOrthogonal):

    def __init__(self, gain=1.0, seed=None, mode='fan_avg'):
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
        return super().__call__(shape, dtype, **kwargs)

    def get_config(self):
        return {'gain': self.gain, 'seed': self.seed, 'mode': self.mode}


class MoreVarianceScaling(VarianceScaling):

    def __init__(self,
                 scale=1.0,
                 mode='fan_in',
                 distribution='truncated_normal',
                 seed=None):
        if scale <= 0.:
            raise ValueError('`scale` must be positive float.')
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument:', mode)
        distribution = distribution.lower()
        # Compatibility with keras-team/keras.
        if distribution == 'normal':
            distribution = 'truncated_normal'
        if distribution not in {'uniform', 'truncated_normal', 'untruncated_normal', 'bi_gamma', 'tanh_normal'}:
            raise ValueError('Invalid `distribution` argument:', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used, which
            default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        scale = self.scale
        fan_in, fan_out = _compute_fans(shape)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)

        if self.distribution == 'truncated_normal':
            # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
            return self._random_generator.truncated_normal(shape, 0.0, stddev, dtype)

        elif self.distribution == 'tanh_normal':
            # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale)
            normal = self._random_generator.random_normal(shape, 0.0, stddev, dtype)
            return tf.math.tanh(normal)

        elif self.distribution == 'bi_gamma':
            # FIXME: without multiplying stddev at the end had very good results
            import numpy as np
            dist = tfd.Gamma(concentration=3.0, rate=2.0)

            # Get 3 samples, returning a 3 x 2 tensor.
            samples = dist.sample(shape)
            flip = 2 * np.random.choice(2, shape) - 1
            samples = samples * flip  # / 10
            stddev = 2 * math.sqrt(scale)
            return stddev * samples

        elif self.distribution == 'untruncated_normal':
            stddev = math.sqrt(scale)
            return self._random_generator.random_normal(shape, 0.0, stddev, dtype)

        else:
            limit = math.sqrt(3.0 * scale)
            return self._random_generator.random_uniform(shape, -limit, limit, dtype)


class GlorotTanh(MoreVarianceScaling):
    def __init__(self, seed=None):
        super().__init__(
            scale=1.0,
            mode='fan_avg',
            distribution='tanh_normal',
            seed=seed)


class BiGamma(MoreVarianceScaling):
    def __init__(self, seed=None):
        super().__init__(
            scale=1.0,
            mode='fan_avg',
            distribution='bi_gamma',
            seed=seed)
