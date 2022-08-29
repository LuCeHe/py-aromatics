import tensorflow as tf

from GenericTools.keras_tools.esoteric_losses import well_loss


class RateVoltageRegularization(tf.keras.layers.Layer):

    def __init__(self, reg_cost, type='', **kwargs):
        super().__init__(**kwargs)
        self.type = type
        self.reg_cost = reg_cost

    def call(self, inputs, training=None):
        b, v_sc = inputs
        rate = tf.reduce_mean(b, 1, keepdims=True)

        if 'rreg' in self.type:
            rate_loss = well_loss(min_value=0, max_value=200 / 1000)(rate) * self.reg_cost,
            # lambda z: util.well_loss(min_value=1 / 1000, max_value=150 / 1000, axis=1)(z) * reg_cost,
            self.add_loss(rate_loss)
            self.add_metric(rate_loss, name='rate_loss_' + self.name, aggregation='mean')

        elif 'rvreg' in self.type:

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
            'type': self.type,
            'reg_cost': self.reg_cost,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
