import tensorflow as tf
import tensorflow.keras.backend as K


# layer

class InputSensitiveGaussianNoise(tf.keras.layers.GaussianNoise):

    def call(self, inputs, training=None):
        # print('training: ', training)

        def noised():
            # print('gaussian')
            # print(self.stddev)
            std = tf.math.reduce_std(inputs)
            return inputs + K.random_normal(
                shape=tf.shape(inputs),
                mean=0.,
                stddev=self.stddev * std,
                dtype=inputs.dtype)

        return K.in_train_phase(noised, inputs, training=training)


# callback

class NoiseSchedule(tf.keras.callbacks.Callback):

    def __init__(self, stddev, epochs):
        self.epochs = epochs
        self.stddev = stddev

    def on_epoch_end(self, epoch, logs=None):
        # print('epoch: ', epoch)
        # std = tf.keras.backend.get_value(self.stddev)
        # print(std)
        if epoch < self.epochs / 2 - 1:
            tf.keras.backend.set_value(self.stddev, 0.)
        else:
            # 0 to .6
            portion = (2 * (epoch + 1) / self.epochs - 1)
            tf.keras.backend.set_value(self.stddev, .6 * portion)
            # print(portion, self.stddev)
        # std = tf.keras.backend.get_value(self.stddev)
        # print(std)
