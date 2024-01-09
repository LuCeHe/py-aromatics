import tensorflow as tf


class GEGLU(tf.keras.layers.Layer):
    def __init__(self, d_h, d_model, activation='sigmoid', comments=''):
        super().__init__()
        # https://arxiv.org/pdf/1612.08083.pdf
        # https://arxiv.org/pdf/2002.05202.pdf

        # model hyper parameter variables
        self.d_model = d_model
        self.d_h = d_h
        self.comments = comments

        if 'noffn' in self.comments:
            self.w_1 = lambda x: x
            self.w_3 = lambda x: x
            self.w_2 = lambda x: x

        elif 'onlyglu' in self.comments:
            self.w_1 = tf.keras.layers.Dense(d_model)
            self.w_3 = tf.keras.layers.Dense(d_model)
            self.w_2 = lambda x: x

        else:
            self.w_1 = tf.keras.layers.Dense(d_h)
            self.w_3 = tf.keras.layers.Dense(d_h)
            self.w_2 = tf.keras.layers.Dense(self.d_model)

        self.activation = tf.keras.layers.Activation(activation)

    def get_config(self):

        config = {
            'd_model': self.d_model,
            'd_h': self.d_h,
            'activation': self.activation,
            'comments': self.comments,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        x1 = self.w_1(inputs)
        x3 = self.w_3(inputs)
        x2 = self.activation(x1) * x3
        return self.w_2(x2)
