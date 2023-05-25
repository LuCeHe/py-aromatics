"""
original article: https://arxiv.org/pdf/1709.07871.pdf
source code: https://stackoverflow.com/questions/55210684/feature-wise-scaling-and-shifting-film-layer-in-keras
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


class FiLM1D(tf.keras.layers.Layer):

    def __init__(self, beta=None, gamma=None, dilation_rate=1, **kwargs):
        self.__dict__.update(dilation_rate=dilation_rate)
        super(FiLM1D, self).__init__(**kwargs)

        random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
        self.beta = self.Conv('beta_' + random_string) if beta == None else beta
        self.gamma = self.Conv('gamma_' + random_string) if gamma == None else gamma

    def Conv(self, name=''):
        return Sequential([
            # Conv1D(16, 3, padding='same', dilation_rate=self.dilation_rate),
            Conv1D(1, 3, padding='same', dilation_rate=self.dilation_rate)
        ], name=name)

    def __call__(self, inputs):
        to_condition, conditioner = inputs

        beta = self.beta(conditioner)
        gamma = self.gamma(conditioner)

        film = gamma * to_condition + beta
        return film


class FiLM_Fusion_2ways(tf.keras.layers.Layer):

    def __init__(self, filters, initializer='orthogonal', kernel_size=3, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)

        self.initializer = initializer
        self.kernel_size = kernel_size
        self.filters = filters

        self.beta1 = Conv1D(self.filters, self.kernel_size, padding='causal', kernel_initializer=self.initializer,
                            dilation_rate=dilation_rate)
        self.gamma1 = Conv1D(self.filters, self.kernel_size, padding='causal', kernel_initializer=self.initializer,
                            dilation_rate=dilation_rate)

        self.beta2 = Conv1D(self.filters, self.kernel_size, padding='causal', kernel_initializer=self.initializer,
                            dilation_rate=dilation_rate)
        self.gamma2 = Conv1D(self.filters, self.kernel_size, padding='causal', kernel_initializer=self.initializer,
                            dilation_rate=dilation_rate)

    def call(self, inputs, **kwargs):
        in1, in2 = inputs

        beta1 = self.beta1(in1)
        gamma1 = self.gamma1(in1)

        beta2 = self.beta2(in2)
        gamma2 = self.gamma2(in2)

        in2 = Add()([Multiply()([in2, gamma1]), beta1])
        in1 = Add()([Multiply()([in1, gamma2]), beta2])

        return (in2, in1)


FiLM_Fusion = FiLM_Fusion_2ways


class FiLM_Fusion_1way(tf.keras.layers.Layer):

    def __init__(self, filters, initializer='orthogonal', kernel_size=3, **kwargs):
        super().__init__(**kwargs)

        self.initializer = initializer
        self.kernel_size = kernel_size
        self.filters = filters

        self.beta = Conv1D(self.filters, self.kernel_size, padding='causal', kernel_initializer=self.initializer)
        self.gamma = Conv1D(self.filters, self.kernel_size, padding='causal', kernel_initializer=self.initializer)

    def call(self, inputs, **kwargs):
        in1, in2 = inputs

        beta = self.beta(in1)
        gamma = self.gamma(in1)

        in2 = Add()([Multiply()([in2, gamma]), beta])

        return in2


if __name__ == '__main__':
    # sound = Input((300, 3))
    # spike = Input((300, 3))
    # filmed_sound = FiLM1D()([sound, spike])
    # filmed_spike = FiLM1D()([spike, sound])
    #
    # o = Concatenate(axis=-1)([filmed_sound, filmed_spike])
    #
    # model = Model([sound, spike], o)
    #
    # model.summary()

    import itertools

    print(list(itertools.permutations('kvq')))
