import tensorflow as tf
import numpy as np


class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, val_data, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # here we use test data to calculate the gradients
        self.multiple_inputs = False
        if isinstance(val_data[0], list):
            self.multiple_inputs = True
            self.splits = len(val_data[0])
            self._x_batch = []
            for v in val_data[0]:
                self._x_batch.append(tf.convert_to_tensor(v, dtype=tf.float32))
            # self._x_batch = tf.convert_to_tensor(self._x_batch)
            self.shapes = [v.shape for v in val_data[0]]
            print(self.shapes)
            flattened_vs = [v.flatten() for v in val_data[0]]
            self.flattened_shapes = [len(v) for v in flattened_vs]
            print(self.flattened_shapes)
            v = np.concatenate(flattened_vs)
            self._x_batch = tf.convert_to_tensor(v)
        else:
            self._x_batch = tf.convert_to_tensor(val_data[0], dtype=tf.float32)
        self._y_batch = tf.convert_to_tensor(val_data[1], dtype=tf.float32)

    def _log_gradients(self, epoch):
        # step = tf.cast(tf.math.floor((epoch + 1) * num_instance / batch_size), dtype=tf.int64)
        writer = self._get_writer(self._train_run_name)

        with writer.as_default(), tf.GradientTape() as g:
            g.watch(self._x_batch)
            if self.multiple_inputs:
                fx = tf.split(self._x_batch, self.flattened_shapes)
                x = [tf.reshape(f, s) for f, s in zip(fx, self.shapes)]
                print(x)
            else:
                x = self._x_batch

            _y_pred = self.model(x)  # forward-propagation
            loss = self.model.loss(y_true=self._y_batch, y_pred=_y_pred)  # calculate loss
            gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(
                    weights.name.replace(':', '_') + '_grads',
                    data=grads, step=epoch)

        writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
        # but we do need to run the original on_epoch_end, so here we use the super function.
        super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)
