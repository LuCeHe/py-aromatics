import collections, csv, io, os

import tensorflow as tf
import numpy as np

from tensorflow.python.lib.io import file_io


# from tensorflow.python.keras.utils.io_utils import path_to_string


# original by DomJack
# https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard


def path_to_string(path):
    """Convert `PathLike` objects to their string representation.

    If given a non-string typed path object, converts it to its string
    representation.

    If the object passed to `path` is not among the above, then it is
    returned unchanged. This allows e.g. passthrough of file objects
    through this function.

    Args:
      path: `PathLike` object that represents a path

    Returns:
      A string representation of the path argument, if Python support exists.
    """
    if isinstance(path, os.PathLike):
        return os.fspath(path)
    return path


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return

        if isinstance(self.model.optimizer.lr, float):
            lr = self.model.optimizer.lr

        elif hasattr(self.model.optimizer.lr, 'numpy') \
                and isinstance(self.model.optimizer.lr.numpy(), (np.floating, float)):
            lr = self.model.optimizer.lr

        elif hasattr(self.model.optimizer.lr, 'lr'):
            lr = self.model.optimizer.lr.lr

        else:
            lr = self.model.optimizer.lr.initial_learning_rate

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
        for n, w in zip(self.w_names, self.logginging_weights):
            logs[n] = w.numpy()


def handle_value(k):
    is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
    if isinstance(k, str):
        return k
    elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
        return '"[%s]"' % (', '.join(map(str, k)))
    else:
        return k


class CSVLogger(tf.keras.callbacks.Callback):
    """Callback that streams epoch results to a CSV file.

    Supports all values that can be represented as a string,
    including 1D iterables such as `np.ndarray`.

    Example:

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    Args:
        filename: Filename of the CSV file, e.g. `'run/log.csv'`.
        separator: String used to separate elements in the CSV file.
        append: Boolean. True: append if file exists (useful for continuing
            training). False: overwrite existing file.
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = path_to_string(filename)
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = ''
        self._open_args = {'newline': '\n'}
        self.pre_epoch_logs = None
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if file_io.file_exists_v2(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            row_dict = collections.OrderedDict({'epoch': -1})
            row_dict.update((key, handle_value(logs[key])) for key in logs.keys())
            self.pre_epoch_logs = row_dict

            # self.on_epoch_end(epoch=-1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict((k, logs[k]) if k in logs else (k, 'NA') for k in self.keys)

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ['epoch'] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)

            if self.append_header:
                self.writer.writeheader()

        if epoch == 0 and not self.pre_epoch_logs is None:
            missing_keys = [k for k in self.keys if not k in self.pre_epoch_logs.keys()]
            self.pre_epoch_logs.update((key, np.nan) for key in missing_keys)
            self.writer.writerow(self.pre_epoch_logs)
            self.csv_file.flush()

        row_dict = collections.OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)

        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
