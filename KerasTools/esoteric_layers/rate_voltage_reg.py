import tensorflow as tf


@tf.custom_gradient
def clip_value_no_grad(x):
    y = tf.clip_by_value(x, -2, 2)

    def custom_grad(dy):
        return dy

    return y, custom_grad


def well_loss(min_value=-120, max_value=40, type='clip_relu_no_clip_grad', axis='all'):
    def wloss(x):
        if type == 'sigmoid':
            loss = -tf.math.sigmoid(x - min_value) + tf.math.sigmoid(x - max_value)
        elif type == 'relu':
            loss = tf.nn.relu(-x + min_value) + tf.nn.relu(x - max_value)
        elif type == 'clip_relu_no_clip_grad':
            loss = tf.nn.relu(-x + min_value) + tf.nn.relu(x - max_value)
            loss = clip_value_no_grad(loss)
        elif type == 'squared':
            loss = tf.square(tf.nn.relu(x - max_value)) + tf.square(tf.nn.relu(min_value - x))
        else:
            raise NotImplementedError

        if axis == 'all':
            return tf.reduce_mean(loss)
        else:
            return tf.reduce_mean(loss, axis=axis)

    return wloss


class RateVoltageRegularization(tf.keras.layers.Layer):

    def __init__(self, reg_cost, type='', **kwargs):
        super().__init__(**kwargs)
        self.type = type
        self.reg_cost = reg_cost

    def call(self, inputs, training=None):
        b, v_sc = inputs
        rate = tf.reduce_mean(b, 1, keepdims=True)

        if 'r_regularization' in self.type:
            rate_loss = well_loss(min_value=0, max_value=200 / 1000)(rate) * self.reg_cost,
            # lambda z: util.well_loss(min_value=1 / 1000, max_value=150 / 1000, axis=1)(z) * reg_cost,
            self.add_loss(rate_loss)
            self.add_metric(rate_loss, name='rate_loss_' + self.name, aggregation='mean')

        elif 'rv_regularization' in self.type:

            max_value = 50 / 1000 if not 'monkey' in self.type else 10 / 1000
            min_value = 5 / 1000 if not 'monkey' in self.type else .2 / 1000
            rate_loss = well_loss(min_value=min_value, max_value=max_value)(rate) * self.reg_cost,
            # lambda z: util.well_loss(min_value=1 / 1000, max_value=150 / 1000, axis=1)(z) * reg_cost,

            volt_loss = well_loss(min_value=-2, max_value=.4)(v_sc) * self.reg_cost,
            # lambda z: util.well_loss(min_value=-3., max_value=1., axis=1)(z) * reg_cost,

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
