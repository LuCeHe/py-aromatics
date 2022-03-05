import tensorflow as tf

from GenericTools.keras_tools.esoteric_layers import SurrogatedStep
from GenericTools.keras_tools.esoteric_layers.rate_voltage_reg import RateVoltageRegularization


@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor, sharpness):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled * sharpness), 0) * dampening_factor
        return [dy * dz_dv_scaled, tf.zeros_like(dampening_factor), tf.zeros_like(sharpness)]

    return tf.identity(z_, name="SpikeFunction"), grad


class LSNN(tf.keras.layers.Layer):
    """
    LSNN
    """

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, units, tau=20., beta=1.8, tau_adaptation=20, dampening_factor=1., ref_period=2., thr=.03,
                 n_regular=0, internal_current=0, initializer='orthogonal', **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(
            units=units, tau=tau, tau_adaptation=tau_adaptation, ref_period=ref_period, thr=thr, n_regular=n_regular,
            dampening_factor=dampening_factor, beta=beta, internal_current=internal_current, initializer=initializer)
        self.__dict__.update(self.init_args)

        self.state_size = (units, units, units, units)
        self.spike_function = SurrogatedStep(dampening=dampening_factor, sharpness=1., config='triangularpseudod')

    def build(self, input_shape):
        n_input = input_shape[-1]
        n = self.units
        self.input_weights = self.add_weight(shape=(n_input, n), initializer=self.initializer, name='input_weights')
        self.recurrent_weights = self.add_weight(shape=(n, n), initializer=self.initializer, name='recurrent_weights')

        self.mask = tf.ones((n, n)) - tf.eye(n)
        self.inh_exc = tf.ones(self.units)
        self._beta = tf.concat([tf.zeros(self.n_regular), tf.ones(n - self.n_regular) * self.beta], axis=0)
        self.built = True

    def refract(self, z, last_spike_distance):
        non_refractory_neurons = tf.cast(last_spike_distance >= self.ref_period, tf.float32)
        z = non_refractory_neurons * z
        new_last_spike_distance = (last_spike_distance + 1) * (1 - z)
        return z, new_last_spike_distance

    def spike(self, new_v, thr, *args):
        v_sc = (new_v - thr) / thr

        # z = SpikeFunction(v_sc, self.dampening_factor, 1.)
        # z.set_shape(v_sc.get_shape())
        z = self.spike_function(v_sc)
        return z, v_sc

    def currents_composition(self, inputs, old_spike):
        external_current = inputs @ self.input_weights

        i_in = external_current + \
               (self.mask * self.recurrent_weights) @ old_spike \
               + self.internal_current

        return i_in

    def threshold_dynamic(self, old_a, old_z):
        decay_a = tf.exp(-1 / self.tau_adaptation)
        new_a = decay_a * old_a + (1 - decay_a) * old_z
        athr = self.thr + new_a * self._beta
        return athr, new_a

    def voltage_dynamic(self, old_v, i_in, i_reset):
        decay = tf.exp(-1 / self.tau)
        new_v = decay * old_v + (1 - decay) * i_in + i_reset
        return new_v

    def call(self, inputs, states, training=None):
        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        old_z, old_v, old_a, last_spike_distance = states

        i_in = self.currents_composition(inputs, old_z)

        thr, new_a = self.threshold_dynamic(old_a, old_z)

        i_reset = - thr * old_z

        new_v = self.voltage_dynamic(old_v, i_in, i_reset)

        z, v_sc = self.spike(new_v, thr, last_spike_distance, old_v, old_a, new_a)

        # refractoriness
        z, new_last_spike_distance = self.refract(z, last_spike_distance)

        output = (z, new_v, thr, v_sc)
        new_state = (z, new_v, new_a, new_last_spike_distance)
        return output, new_state


if __name__ == '__main__':
    cell = LSNN(units=2)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, name='encoder')
    # reg = RateVoltageRegularization(.1, type='rv_regularization')

    t = tf.random.uniform((2, 3, 4))

    # ==================================================================
    # check spiking activity
    # ==================================================================

    input_layer = tf.keras.layers.Input(t.shape[1:])
    z, _, _, v_sc = rnn(input_layer)
    # z = reg([z, v_sc])
    model = tf.keras.models.Model(input_layer, z)

    prediction = model.predict(t)
    print(prediction)

    # ==================================================================
    # check .fit
    # ==================================================================

    input_layer = tf.keras.layers.Input(t.shape[1:])
    z, _, _, v_sc = rnn(input_layer)
    output = tf.keras.layers.Dense(t.shape[-1])(z)
    # z = reg([z, v_sc])
    model = tf.keras.models.Model(input_layer, output)
    model.compile('SGD', 'mse')

    model.fit(t, t)
