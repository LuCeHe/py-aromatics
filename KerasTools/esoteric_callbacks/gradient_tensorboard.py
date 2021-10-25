import tensorflow as tf
import numpy as np
from tensorflow.python.ops import summary_ops_v2


def _log_grads(self, epoch):
    with tf.GradientTape(persistent=True) as tape:
        # This capture current state of weights
        tape.watch(self.model.trainable_weights)

        # Calculate loss for given current state of weights
        _y_pred = self.model(self._x_batch)
        # loss = self.model.compiled_loss(y_true=self._y_batch, y_pred=_y_pred)
        loss = self.model.compiled_loss(
            y_true=self._y_batch, y_pred=_y_pred, sample_weight=None, regularization_losses=self.model.losses
        )

    # Calculate Grads wrt current weights
    grads = [tape.gradient(loss, l.trainable_weights) for l in self.model.layers]
    names = [l.name for l in self.model.layers]
    del tape

    with self._train_writer.as_default():

        with summary_ops_v2.always_record_summaries():
            for g, n in zip(grads, names):
                if len(g) > 0:
                    for i, curr_grad in enumerate(g):
                        if len(curr_grad) > 0:
                            nc = 'bias' if len(curr_grad.shape) == 1 else 'weight'
                            # curr_grad(g)
                            mean = tf.reduce_mean(tf.abs(curr_grad))
                            # tf.summary.scalar('grad_mean_{}_{}'.format(n, i + 1), mean)
                            summary_ops_v2.scalar('grad_mean_{}_{}_{}'.format(n, i + 1, nc), mean, step=epoch)
                            # tf.summary.histogram('grad_histogram_{}_{}'.format(n, i + 1), curr_grad)
                            summary_ops_v2.histogram('grad_histogram_{}_{}_{}'.format(n, i + 1, nc), curr_grad,
                                                     step=epoch)

    self._train_writer.flush()


def _log_weights_individual(self, epoch):
    """Logs the weights of the Model to TensorBoard."""
    if epoch == 0:
        self.dict_scalar_locations = {}
    with self._train_writer.as_default():
        with summary_ops_v2.always_record_summaries():
            for layer in self.model.layers:
                for weight in layer.weights:
                    weight_name = weight.name.replace(':', '_')
                    summary_ops_v2.histogram(weight_name, weight, step=epoch)

                    # I add these 3 lines to record some of the weights individually
                    for i in range(self.n_individual_weight_samples):
                        scalar_name = '{}_{}'.format(weight.name.replace(':', '_'), i)
                        if epoch == 0:
                            c = [np.random.choice(ax) for ax in weight.shape]
                            self.dict_scalar_locations[scalar_name] = c
                        else:
                            c = self.dict_scalar_locations[scalar_name]
                        summary_ops_v2.scalar(scalar_name, weight[c], step=epoch)

                    if self.write_images:
                        self._log_weight_as_image(weight, weight_name, epoch)
            self._train_writer.flush()


class GradientTensorBoard(tf.keras.callbacks.TensorBoard):
    # https://medium.com/@leenabora1/how-to-keep-a-track-of-gradients-vanishing-exploding-gradients-b0bbaa1dcb93
    def __init__(self, validation_data, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # here we use test data to calculate the gradients
        self._x_batch = validation_data[0]
        self._y_batch = validation_data[1] if len(validation_data) == 2 else None

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_grads(epoch)


class IndividualWeightsTensorBoard(tf.keras.callbacks.TensorBoard):
    n_individual_weight_samples = 5

    def _log_weights(self, epoch):
        _log_weights_individual(self, epoch)


class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
    # https://medium.com/@leenabora1/how-to-keep-a-track-of-gradients-vanishing-exploding-gradients-b0bbaa1dcb93
    def __init__(self, validation_data, n_individual_weight_samples=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # here we use test data to calculate the gradients
        self._x_batch = validation_data[0]
        self._y_batch = validation_data[1] if len(validation_data) == 2 else None
        self.n_individual_weight_samples = n_individual_weight_samples

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            _log_grads(self, epoch)

    def _log_weights(self, epoch):
        _log_weights_individual(self, epoch)
