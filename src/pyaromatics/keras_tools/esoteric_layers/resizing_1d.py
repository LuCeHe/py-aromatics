import tensorflow as tf


class Resizing1D(tf.keras.layers.Layer):

    def __init__(self, time_steps, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.time_steps, self.feature_dim = time_steps, feature_dim
        self.resize = tf.keras.layers.experimental.preprocessing.Resizing(
            time_steps, feature_dim, interpolation="bilinear", **kwargs
        )

    def call(self, inputs, training=None):
        inputs = tf.expand_dims(inputs, axis=-1)
        outputs = self.resize(inputs)[..., 0]
        return outputs

    def get_config(self):
        config = {
            'time_steps': self.time_steps, 'feature_dim': self.feature_dim
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    t = tf.random.uniform((2, 3, 3))
    o = Resizing1D(7, 7)(t)
    print(o.shape)
