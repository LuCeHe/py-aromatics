import tensorflow as tf
import numpy as np
from tensorflow.python.ops import summary_ops_v2


class GradientTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, val_data, track_operation_on, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # here we use test data to calculate the gradients
        self.multiple_inputs = False
        if isinstance(val_data[0], tuple):
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

        self.track_operation_on = track_operation_on

    def _log_track_operation_on(self, epoch):
        for v_name, op in self.track_operation_on:
            try:
                weights = [v for v in self.model.trainable_weights if v_name in v.name][0]
                operated_weights = op(weights)
                tf.summary.scalar("op on {}".format(v_name), operated_weights, step=epoch)
            except Exception as e:
                print(e)

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

    def _log_single_weights(self, epoch):
        # step = tf.cast(tf.math.floor((epoch + 1) * num_instance / batch_size), dtype=tf.int64)
        writer = self._get_writer(self._train_run_name)

        with writer.as_default():
            for v in self.model.trainable_weights:
                if len(v.shape) == 2:
                    for _ in range(10):
                        i = np.random.choice(v.shape[0])
                        j = np.random.choice(v.shape[1])
                        tf.summary.scalar(v.name + "_single_weight_{}_{}".format(i, j),
                                          v[i, j], step=epoch)

        writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
        # but we do need to run the original on_epoch_end, so here we use the super function.
        super().on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)
            self._log_single_weights(epoch)


class IndividualWeightsTensorBoard(tf.keras.callbacks.TensorBoard):

    def _log_weights(self, epoch):
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
                        for i in range(5):
                            scalar_name = '{}_{}'.format(weight.name.replace(':', '_'), i)
                            if epoch == 0:
                                c = [np.random.choice(ax) for ax in weight.shape]
                                self.dict_scalar_locations[scalar_name] = c
                            else:
                                c = self.dict_scalar_locations[scalar_name]
                            summary_ops_v2.scalar(scalar_name, weight[c], step=epoch)
                            # summary_ops_v2.add_summary(scalar_name, weight[c], step=epoch)
                            # summary_ops_v2.scalar(weight.name, weight[c], step=epoch)

                        if self.write_images:
                            self._log_weight_as_image(weight, weight_name, epoch)
                self._train_writer.flush()
