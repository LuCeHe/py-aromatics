import tensorflow as tf

from GenericTools.keras_tools.esoteric_losses import get_loss


class AddMetricsLayer(tf.keras.layers.Layer):

    def __init__(self, metrics, prefix_id='', **kwargs):
        super().__init__(**kwargs)
        self.additional_metrics = metrics
        # self.additional_metrics = [get_loss(l) if isinstance(l,str) else l for l in self.additional_metrics]
        self.prefix_id = prefix_id

    def call(self, inputs, **kwargs):
        true_output, pred_output = inputs

        for m in self.additional_metrics:
            m_output = m(true_output, pred_output)
            name = m.name if hasattr(m, 'name') else m.__name__
            self.add_metric(m_output, name=self.prefix_id + name, aggregation='mean')
        return pred_output

    def get_config(self):
        config = {
            'metrics': self.additional_metrics,
            'prefix_id': self.prefix_id,
        }
        return dict(list(super().get_config().items()) + list(config.items()))
