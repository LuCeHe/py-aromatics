# original: https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras

import tensorflow as tf


class MultipleValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0):
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

    def on_epoch_end(self, epoch, logs=None):

        if logs is None:
            return

        # evaluate on the additional validation sets
        for val_name, generator in self.validation_sets.items():
            evaluation = self.model.evaluate(generator, return_dict=True, verbose=False)

            for k in evaluation.keys():
                logs[val_name + '_' + k] = evaluation[k]

        # print(logs)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            if logs is None:
                return

            # evaluate on the additional validation sets
            for val_name, generator in self.validation_sets.items():
                evaluation = self.model.evaluate(generator, return_dict=True, verbose=False)

                for k in evaluation.keys():
                    logs[val_name + '_' + k + '_begin'] = evaluation[k]
