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
        logs["learning_rate"] = lr.numpy()