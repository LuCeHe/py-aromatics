import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.initializers.initializers_v2 import _RandomGenerator, _compute_fans, Orthogonal

tfd = tfp.distributions
_PARTITION_SHAPE = 'partition_shape'


def orthogonalize(initial_initializer):
    shape = initial_initializer.shape

    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))

    a = tf.reshape(initial_initializer, flat_shape)
    # Compute the qr factorization
    q, r = tf.linalg.qr(a, full_matrices=False)
    # Make Q uniform
    d = tf.linalg.tensor_diag_part(r)
    q *= tf.sign(d)
    if num_rows < num_cols:
        q = tf.linalg.matrix_transpose(q)
    orthogonal_initializer = tf.reshape(q, shape)
    return orthogonal_initializer


class MoreVarianceScalingAndOrthogonal(tf.keras.initializers.Initializer):

    def get_config(self):
        return {
            'scale': self.scale,
            'mode': self.mode,
            'distribution': self.distribution,
            'seed': self.seed,
            'orthogonalize': self.orthogonalize
        }

    def __init__(self,
                 scale=1.0,
                 mode='fan_in',
                 distribution='truncated_normal',
                 orthogonalize=False,
                 seed=None):
        if scale <= 0.:
            raise ValueError('`scale` must be positive float.')
        if mode not in {'fan_in', 'fan_out', 'fan_avg', 'no_fan'}:
            raise ValueError('Invalid `mode` argument:', mode)
        distribution = distribution.lower()
        # Compatibility with keras-team/keras.
        if distribution == 'normal':
            distribution = 'truncated_normal'
        distributions_possible = ['uniform', 'truncated_normal', 'untruncated_normal', 'bi_gamma', 'bi_gamma_10',
                                  'tanh_normal', 'cauchy', 'nozero_uniform']
        tanh_distributions = ['tanh_' + d for d in distributions_possible]
        distributions_possible.extend(tanh_distributions)
        if distribution not in distributions_possible:
            raise ValueError('Invalid `distribution` argument:', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.orthogonalize = orthogonalize
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=tf.float32, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used, which
            default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        """
        # _validate_kwargs(self.__class__.__name__, kwargs)
        # dtype = _assert_float_dtype(_get_dtype(dtype))

        scale = self.scale
        fan_in, fan_out = _compute_fans(shape)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        elif self.mode == 'no_fan':
            pass
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)

        if 'untruncated_normal' in self.distribution:
            stddev = tf.sqrt(scale)
            distribution = self._random_generator.random_normal(shape, 0.0, stddev, dtype)

        elif 'truncated_normal' in self.distribution:
            # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = tf.sqrt(scale) / .87962566103423978
            distribution = self._random_generator.truncated_normal(shape, 0.0, stddev, dtype)

        elif 'bi_gamma_10' in self.distribution:
            dist = tfd.Gamma(concentration=10.0, rate=10.0)
            samples = dist.sample(shape)
            flip = 2 * np.random.choice(2, shape) - 1
            stddev = 2 * tf.sqrt(scale)
            distribution = stddev * samples * flip / 10

        elif 'bi_gamma' in self.distribution:
            alpha = 3
            beta = 2
            dist = tfd.Gamma(concentration=alpha, rate=beta)
            samples = dist.sample(shape)
            flip = 2 * np.random.choice(2, shape) - 1
            stddev = tf.sqrt((beta ** 2) / (alpha * (alpha + 1)))
            distribution = stddev * samples * flip
            stddev = tf.sqrt(scale)
            distribution = stddev * distribution

        elif 'cauchy' in self.distribution:
            dist = tfd.Cauchy(loc=0., scale=1.)
            stddev = tf.sqrt(scale) / 2
            distribution = stddev * dist.sample(shape)

        elif 'nozero_uniform' in self.distribution:
            stddev = tf.sqrt(3.0 * scale)
            dist = tfd.Uniform(low=[-1.0, .25], high=[-.25, 1])
            distribution = stddev * dist.sample(shape)
            distribution = tf.reshape(distribution, (-1))
            distribution = tf.random.shuffle(distribution)
            distribution = tf.reshape(distribution, (*shape, 2))[..., 0]

        else:
            stddev = tf.sqrt(3.0 * scale)
            distribution = self._random_generator.random_uniform(shape, -stddev, stddev, dtype)

        if 'tanh' in self.distribution:
            distribution = stddev * tf.math.tanh(distribution)

        if self.orthogonalize:
            # std = tf.math.reduce_std(distribution)
            distribution = orthogonalize(distribution)
            # new_std = tf.math.reduce_std(distribution)
            # distribution = distribution*std/new_std

        return distribution


class GlorotTanh(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='fan_avg', distribution='tanh_normal', seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)


class BiGamma(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='no_fan', distribution='bi_gamma', seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)


class GlorotBiGamma(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='fan_avg', distribution='bi_gamma', seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)


class HeBiGamma(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=2.0, mode='fan_in', distribution='bi_gamma', seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)


class GlorotBiGammaOrthogonal(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='fan_avg', distribution='bi_gamma', orthogonalize=True, seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed, orthogonalize=orthogonalize)


class BiGammaOrthogonal(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='no_fan', distribution='bi_gamma', orthogonalize=True, seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed, orthogonalize=orthogonalize)


class BiGamma10(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='fan_avg', distribution='bi_gamma_10', seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)


class TanhBiGamma10(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='fan_avg', distribution='tanh_bi_gamma_10', seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)


class TanhBiGamma10Orthogonal(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='fan_avg', distribution='tanh_bi_gamma_10', orthogonalize=True, seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed, orthogonalize=orthogonalize)


class GlorotCauchyOrthogonal(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, seed=None):
        super().__init__(scale=scale, mode='fan_avg', distribution='cauchy', orthogonalize=True, seed=seed)


class CauchyOrthogonal(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='no_fan', distribution='cauchy', orthogonalize=True, seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed, orthogonalize=orthogonalize)


class GlorotOrthogonal(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='fan_avg', distribution='uniform', orthogonalize=True, seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed, orthogonalize=orthogonalize)


class NoZeroGlorotOrthogonal(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='fan_avg', distribution='nozero_uniform', orthogonalize=True, seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed, orthogonalize=orthogonalize)


class NoZeroGlorot(MoreVarianceScalingAndOrthogonal):
    def __init__(self, scale=1.0, mode='fan_avg', distribution='nozero_uniform', orthogonalize=False, seed=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed, orthogonalize=orthogonalize)


if __name__ == '__main__':
    initializer = MoreVarianceScalingAndOrthogonal(
        scale=1.0,
        mode='no_fan',
        distribution='bi_gamma',  # 'tanh_bi_gamma',
        orthogonalize=True,
        seed=None
    )

    shape = (20, 30)
    t = initializer(shape).numpy()

    tt_1 = np.dot(t, t.T)
    product_1 = np.abs(tt_1) - np.eye(tt_1.shape[-1])
    # tt_2 = np.dot(t.T, t)
    # product_2 = np.abs(tt_2) - np.eye(tt_2.shape[-1])

    # print(product_2)
    print('{}Orthogonal! '.format('' if np.all(product_1 < 1e-6) else 'Not '))
    print('Variance: ', np.std(t) ** 2)
    import matplotlib.pyplot as plt

    n, bins, patches = plt.hist(x=t.flatten(), bins=50, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.show()