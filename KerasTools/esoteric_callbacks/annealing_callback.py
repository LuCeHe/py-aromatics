import tensorflow as tf

Callback = tf.keras.callbacks.Callback


class AnnealingCallback(Callback):

    def __init__(self,
                 epochs,
                 variables_to_anneal=[],
                 histogram_freq=0,
                 **kwargs):
        super().__init__()
        self.epochs = epochs
        self.histogram_freq = histogram_freq
        self.variables_to_anneal = variables_to_anneal

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
        self.model = model
        self.annealing_weights = []
        for layer in self.model.layers:
            for weight in layer.weights:
                for va in self.variables_to_anneal:
                    if va in weight.name:
                        self.annealing_weights.append(weight)

    def on_epoch_begin(self, epoch, logs=None):

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            if epoch >= self.epochs / 2:
                for w in self.annealing_weights:
                    v = tf.keras.backend.get_value(w)
                    tf.keras.backend.set_value(w, .6 * v)
