import tensorflow as tf

from pyaromatics.keras_tools.esoteric_losses import well_loss
from pyaromatics.stay_organized.utils import str2val


class RateVoltageRegularization(tf.keras.layers.Layer):

    def __init__(self, reg_cost=1., config='', **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.reg_cost = reg_cost

        self.ff_switch = 0.
        self.target_firing_rate = str2val(self.config, 'adjff', float, default=.1)

        if 'heidelberg' in config:
            self.reg_cost = 0.436
        elif 'wordptb' in config:
            self.reg_cost = 4.595
        elif 'sl_mnist' in config:
            self.reg_cost = 0.800

    def build(self, input_shape):
        stacki = str2val(self.config, 'stacki', int, default=0)
        self.coef = self.add_weight(name=f'switch_{stacki}',
                                    shape=(),
                                    initializer=tf.keras.initializers.Constant(self.ff_switch),
                                    trainable=False)

        self.built = True

    def call(self, inputs, training=None):
        b, v_sc = inputs
        rate = tf.reduce_mean(b, 1, keepdims=True)

        if 'rreg' in self.config:
            rate_loss = well_loss(min_value=0, max_value=200 / 1000)(rate) * self.reg_cost,
            # lambda z: util.well_loss(min_value=1 / 1000, max_value=150 / 1000, axis=1)(z) * reg_cost,
            self.add_loss(rate_loss)
            self.add_metric(rate_loss, name='rate_loss_' + self.name, aggregation='mean')

        elif 'adjff' in self.config:
            rate_loss = well_loss(
                min_value=self.target_firing_rate, max_value=self.target_firing_rate, walls_type='relu'
            )(rate)
            rate_loss = rate_loss * self.reg_cost * self.coef

            self.add_loss(rate_loss)
            self.add_metric(rate_loss, name='rate_loss_' + self.name, aggregation='mean')

        elif 'rvreg' in self.config:

            max_value = 50 / 1000 if not 'monkey' in self.type else 10 / 1000
            min_value = 5 / 1000 if not 'monkey' in self.type else .2 / 1000
            rate_loss = well_loss(min_value=min_value, max_value=max_value)(rate) * self.reg_cost,

            volt_loss = well_loss(min_value=-2, max_value=.4)(v_sc) * self.reg_cost,

            self.add_loss(rate_loss)
            self.add_metric(rate_loss, name='rate_loss_' + self.name, aggregation='mean')
            self.add_loss(volt_loss)
            self.add_metric(volt_loss, name='volt_loss_' + self.name, aggregation='mean')

        return b

    def get_config(self):
        config = {
            'config': self.config,
            'reg_cost': self.reg_cost,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
