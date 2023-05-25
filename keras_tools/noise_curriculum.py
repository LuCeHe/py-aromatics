import tensorflow as tf
import tensorflow.keras.backend as K


# layer

class InputSensitiveGaussianNoise(tf.keras.layers.GaussianNoise):

    def build(self, input_shape):
        self.stddev = self.add_weight(shape=(), initializer="zeros", trainable=False,
                                      name='curriculum_noise')

    def call(self, inputs, training=None):
        if not training is None:
            tf.keras.backend.set_learning_phase(training)
        is_train = tf.cast(tf.keras.backend.learning_phase(), tf.float32)

        std = tf.math.reduce_std(inputs)
        output = inputs + is_train * self.stddev * std * tf.random.normal(tf.shape(inputs))

        return output


# callback

class NoiseSchedule(tf.keras.callbacks.Callback):

    def __init__(self, stddev, epochs):
        self.epochs = epochs
        self.stddev = stddev

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.epochs / 4 - 1 or epoch > 3*self.epochs / 4 - 1:
            tf.keras.backend.set_value(self.stddev, 0.)
        else:
            # 0 to .6
            portion = (2 * (epoch + 1) / self.epochs - 1)
            tf.keras.backend.set_value(self.stddev, .4 * portion)
