import tensorflow as tf

from pyaromatics.keras_tools.esoteric_losses import get_loss


class AddLossLayer(tf.keras.layers.Layer):

    def __init__(self, loss, coef=1., prefix_id='', aggregation='mean', **kwargs):
        super().__init__(**kwargs)
        self.loss = loss
        self.aggregation = aggregation
        self.coef = coef
        self.losses_to_add = loss if isinstance(loss, list) else [loss]
        self.losses_to_add = [get_loss(l) if isinstance(l, str) else l for l in self.losses_to_add]

        self.coefs = [coef for _ in self.losses_to_add]
        self.prefix_id = prefix_id

        self.loss_names = []
        for l in self.losses_to_add:
            try:
                n = l.name
            except:
                try:
                    n = l.__name__
                except:
                    n = str(l)
            self.loss_names.append(self.prefix_id + n)

    def build(self, input_shape):

        if len(self.losses_to_add) > 1:
            coefs = []
            for i, ci in enumerate(list(self.coefs)):
                c = self.add_weight(
                    name='loss_' + str(i), shape=(), initializer=tf.keras.initializers.Constant(ci),
                    trainable=False
                )
                coefs.append(c)
            self.coefs = coefs

        self.built = True

    def call(self, inputs, training=None):
        true_output, pred_output = inputs

        for c, l, n in zip(self.coefs, self.losses_to_add, self.loss_names):
            loss = tf.reduce_mean(c * l(true_output, pred_output))
            self.add_loss(loss)
            self.add_metric(loss, name=n, aggregation=self.aggregation)

        return pred_output

    def get_config(self):
        config = {
            'coef': self.coef,
            'loss': self.loss,
            'prefix_id': self.prefix_id,
            'aggregation': self.aggregation,
        }
        return dict(list(super().get_config().items()) + list(config.items()))
