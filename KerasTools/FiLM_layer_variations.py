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

    def __init__(self, beta=None, gamma=None, dilation_rate=2, **kwargs):
        self.__dict__.update(dilation_rate=dilation_rate)
        super(FiLM1D, self).__init__(**kwargs)

        random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
        self.beta = self.Conv('beta_' + random_string) if beta == None else beta
        self.gamma = self.Conv('gamma_' + random_string) if gamma == None else gamma

    def Conv(self, name=''):
        return Sequential([
            Conv1D(16, 3, dilation_rate=self.dilation_rate),
            Conv1D(1, 3, dilation_rate=self.dilation_rate)
        ], name=name)

    def __call__(self, inputs):
        to_condition, conditioner = inputs

        beta = self.beta(conditioner)
        gamma = self.gamma(conditioner)

        film = gamma * to_condition + beta
        return film


if __name__ == '__main__':
    sound = Input((None, 3))
    spike = Input((None, 3))
    filmed_sound = FiLM1D()([sound, spike])
    filmed_spike = FiLM1D()([spike, sound])

    o = Concatenate(axis=-1)([filmed_sound, filmed_spike])

    model = Model([sound, spike], o)

    model.summary()
