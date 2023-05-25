# make a dropout keras layer that cycles on and off every epoch

# this is useful for training a model with a large batch size

import tensorflow as tf
import numpy as np


class CyclingDropout(tf.keras.layers.Layer):

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, high_p, low_p=0, freq=1, **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(high_p=high_p, low_p=low_p, freq=freq)
        self.__dict__.update(self.init_args)
        self.state_size = (1,)

    def build(self, input_shape):

        random_string = ''.join([str(r) for r in np.random.choice(10, 4)])

        self.state = self.add_weight(shape=(), initializer='zeros', trainable=False,
                                     name=f'cyclingdropout_state_{random_string}')

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):

        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        if tf.keras.backend.learning_phase():

            # cosine between high_p, low_p
            p = self.high_p + (self.low_p - self.high_p) * (1 + tf.cos(self.freq * self.state * 3.14159 / 2)) / 2
            n = tf.random.uniform(shape=tf.shape(inputs), minval=0, maxval=1, dtype=tf.float32) > p
            n = tf.cast(n, tf.float32)

            output = inputs * n
        else:
            output = inputs
        return output


class rnnCyclingDropout(tf.keras.layers.Layer):

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, high_p, low_p=0, freq=1, **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(high_p=high_p, low_p=low_p, freq=freq)
        self.__dict__.update(self.init_args)
        self.state_size = (1,)

    def call(self, inputs, states, training=None):
        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        if tf.keras.backend.learning_phase():
            print('here!', states[0])
            state = states[0] + 1

            # cosine between high_p, low_p
            p = self.high_p + (self.low_p - self.high_p) * (1 + tf.sin(self.freq * state * 3.14159 / 2) / 2)
            n = tf.random.uniform(shape=tf.shape(inputs), minval=0, maxval=1, dtype=tf.float32) > p
            n = tf.cast(n, tf.float32)

            output = inputs * n
            new_state = (state,)
        else:
            output = inputs
            new_state = states
        return output, new_state


# make the previous into an RNN layer
class CyclingDropoutRNN(tf.keras.layers.RNN):
    def __init__(self, high_p, low_p=0, freq=1, **kwargs):
        cell = rnnCyclingDropout(high_p, low_p, freq, **kwargs)
        super().__init__(cell, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.cell.init_args.items()))


if __name__ == '__main__':
    from tqdm import tqdm

    shape = (2, 50, 100)
    t = tf.ones(shape)
    r = CyclingDropout(.5, .2, .1)

    om = []

    for i in tqdm(range(100)):
        # print('-' * 20)
        # print(i)
        r.state = i
        o = r(t, training=True)
        om.append(o.numpy().mean())

    import matplotlib.pyplot as plt

    plt.plot(om)
    plt.show()
