# original: https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras

import tensorflow as tf


class MultipleValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0, record_before_training=True):
        """
        :param validation_sets:
        a dictionary with the name of the validation set as key, and the generator of the validation set as the value
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super().__init__()
        self.validation_sets = validation_sets
        self.verbose = verbose
        self.record_before_training = record_before_training

    def on_epoch_end(self, epoch, logs=None):

        if logs is None:
            logs = {}

        # evaluate on the additional validation sets
        for val_name, generator in self.validation_sets.items():
            evaluation = self.model.evaluate(generator, return_dict=True, verbose=self.verbose)

            for k in evaluation.keys():
                logs[val_name + '_' + k] = evaluation[k]

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0 and self.record_before_training:
            self.on_epoch_end(epoch, logs=logs)
