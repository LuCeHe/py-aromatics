import tensorflow as tf

# original by DomJack
# https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        if isinstance(self.model.optimizer.lr, float):
            lr = self.model.optimizer.lr
        else:
            lr = self.model.optimizer.lr(epoch)
        # print(lr)
        logs["learning_rate"] = lr.numpy() if hasattr(lr, 'numpy') else lr



class VariablesLogger(tf.keras.callbacks.Callback):
    def __init__(self, variables_to_log=None):
        super().__init__()
        self._supports_tf_logs = True
        self.variables_to_log = variables_to_log

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
        self.model = model

        if not self.variables_to_log is None:
            self.logginging_weights = []
            self.w_names = []

            for va in self.variables_to_log:
                for layer in self.model.layers:
                    for weight in layer.weights:
                        if va in weight.name:
                            self.logginging_weights.append(weight)
                            self.w_names.append(weight.name)

    def on_epoch_end(self, epoch, logs=None):
        for n, w in zip(self.w_names, self.logginging_weights) :
            logs[n] = w.numpy()