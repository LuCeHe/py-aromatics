import tensorflow as tf

class AddLossLayer(tf.keras.layers.Layer):

    def __init__(self, loss, coef=1., **kwargs):
        super().__init__(**kwargs)
        self.coef = coef
        self.loss = loss

        try:
            self.loss_name = self.loss.name
        except:
            self.loss_name = self.loss.__name__

    def call(self, inputs, training=None):
        true_output, pred_output = inputs

        loss = tf.reduce_sum(self.coef * self.loss(true_output, pred_output))
        self.add_loss(loss)
        self.add_metric(loss, name=self.loss_name, aggregation='mean')
        return pred_output


    def get_config(self):
        config = {
            'coef': self.coef,
            'loss': self.loss,
        }
        return dict(list(super().get_config().items()) + list(config.items()))
