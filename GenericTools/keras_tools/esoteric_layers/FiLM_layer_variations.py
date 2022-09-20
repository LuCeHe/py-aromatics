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
            #Conv1D(16, 3, padding='same', dilation_rate=self.dilation_rate),
            Conv1D(1, 3, padding='same', dilation_rate=self.dilation_rate)
        ], name=name)

    def __call__(self, inputs):
        to_condition, conditioner = inputs

        beta = self.beta(conditioner)
        gamma = self.gamma(conditioner)

        film = gamma * to_condition + beta
        return film


def FiLM_Fusion(size, data_type='FiLM_v2', initializer='orthogonal'):
    def fuse(inputs):
        sound, spikes = inputs
        if 'FiLM_v1' in data_type or 'FiLM_v2' in data_type:
            # FiLM starts -------
            beta_snd = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(spikes)
            gamma_snd = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(spikes)

            beta_spk = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(sound)
            gamma_spk = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(sound)

            # sound = Multiply()([sound, gamma_snd]) + beta_snd
            sound = Add()([Multiply()([sound, gamma_snd]), beta_snd])
            # spikes = Multiply()([spikes, gamma_spk]) + beta_spk
            spikes = Add()([Multiply()([spikes, gamma_spk]), beta_spk])

            # FiLM ends ---------

            layer = [sound, spikes]

        elif 'FiLM_v3' in data_type or 'FiLM_v4' in data_type:
            # FiLM starts -------
            beta_snd = Dense(size)(spikes)
            gamma_snd = Dense(size)(spikes)

            beta_spk = Dense(size)(sound)
            gamma_spk = Dense(size)(sound)
            # changes: 20-8-20 instead of + I made a layer with ADD

            # sound = Multiply()([sound, gamma_snd]) + beta_snd
            sound = Add()([Multiply()([sound, gamma_snd]), beta_snd])
            # spikes = Multiply()([spikes, gamma_spk]) + beta_spk
            spikes = Add()([Multiply()([spikes, gamma_spk]), beta_spk])

            # FiLM ends ---------

            layer = [sound, spikes]

        elif 'FiLM_v5' in data_type:

            # just modulating the sound
            # FiLM starts -------
            beta_snd = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(spikes)
            gamma_snd = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(spikes)

            # changes: 20-8-20 instead of + I made a layer with ADD
            # sound = Multiply()([sound, gamma_snd]) + beta_snd
            sound = Add()([Multiply()([sound, gamma_snd]), beta_snd])
            # spikes = Multiply()([spikes, gamma_spk]) + beta_spk

            # FiLM ends ---------

            layer = [sound, spikes]

        elif 'FiLM_v6' in data_type:
            # FiLM starts -------

            # just modulating the spikes
            beta_spk = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(sound)
            gamma_spk = Conv1D(size, 3, padding='same', kernel_initializer=initializer)(sound)

            # changes: 20-8-20 instead of + I made a layer with ADD
            # sound = Multiply()([sound, gamma_snd]) + beta_snd
            # spikes = Multiply()([spikes, gamma_spk]) + beta_spk
            spikes = Add()([Multiply()([spikes, gamma_spk]), beta_spk])

            # FiLM ends ---------

            layer = [sound, spikes]

        else:
            layer = inputs
        return layer

    return fuse

if __name__ == '__main__':
    sound = Input((300, 3))
    spike = Input((300, 3))
    filmed_sound = FiLM1D()([sound, spike])
    filmed_spike = FiLM1D()([spike, sound])

    o = Concatenate(axis=-1)([filmed_sound, filmed_spike])

    model = Model([sound, spike], o)

    model.summary()
