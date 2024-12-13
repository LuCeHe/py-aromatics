
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class CumulativeGumbel(Layer):
    def __init__(self, mu=0., beta=1., name='CumulativeGumbel', **kwargs):
        self.mu = mu
        self.beta = beta
        super(CumulativeGumbel, self).__init__(**kwargs, name=name)

    def call(self, inputs):
        K.mean(inputs, axis=self.axis)
        exp_argument = -(inputs - self.mu) / self.beta
        log_c_gumbel = -K.exp(exp_argument)
        c_gumbel = K.exp(log_c_gumbel)
        return [c_gumbel, log_c_gumbel]

    def compute_output_shape(self, input_shape):
        return input_shape, input_shape


class GumbelSoftmax(Layer):
    """Softmax activation function.

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as the input.

    Arguments:
      axis: Integer, axis along which the softmax normalization is applied.
    """

    def __init__(self, axis=-1, gumbel_temperature=1., **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.gumbel_temperature = K.variable(gumbel_temperature)

    def call(self, inputs):
        g_noise = gumbel_noise(tf.shape(inputs))

        noisy = (K.log(inputs)+g_noise)/self.gumbel_temperature
        return K.softmax(noisy, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def gumbel_noise(shape):
    uniform_noise = tf.random.uniform(shape)
    return -K.log(-K.log(uniform_noise))



class GumbelTemperatureAnnealing(tf.keras.callbacks.Callback):

    def __init__(self, temperature, min_temperature=.1, max_temperature=10., epoch_start=0, epoch_end = 2):
        self.gumbel_temperature = temperature

        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.epoch = 0

    # customize your behavior
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    # customize your behavior
    def on_batch_end(self, epoch, logs={}):

        new_gt = K.get_value(self.gumbel_temperature)
        if self.epoch == self.epoch_start:
            new_gt = max(new_gt - .01, self.min_temperature)
        elif self.epoch == self.epoch_end:
            new_gt = self.min_temperature

        K.set_value(self.gumbel_temperature, K.constant(new_gt))

        # logger.info(" epoch %s, alpha = %s, beta = %s" % (epoch, K.get_value(self.alpha), K.get_value(self.beta)))
