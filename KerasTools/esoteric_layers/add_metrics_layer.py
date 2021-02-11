import tensorflow as tf

class AddMetricsLayer(tf.keras.layers.Layer):

    def __init__(self, metrics, **kwargs):
        super().__init__(**kwargs)
        self.additional_metrics = metrics

    def call(self, inputs, training=None):
        true_output, pred_output = inputs

        for m in self.additional_metrics:
            m_output = m(true_output, pred_output)
            name = m.name if hasattr(m, 'name') else m.__name__
            self.add_metric(m_output, name=name, aggregation='mean')
        return pred_output


    def get_config(self):
        config = {
            'metrics': self.additional_metrics,
        }
        return dict(list(super().get_config().items()) + list(config.items()))
