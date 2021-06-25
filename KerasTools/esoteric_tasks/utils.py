import numpy as np
import tensorflow as tf
from GenericTools.KerasTools.reinitialize import reset_weights


def unpackbits(x, num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = 2 ** np.arange(num_bits).reshape([1, num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])


class FreezeAndReinitialize(tf.keras.callbacks.Callback):
    def __init__(self, model=None, when_far=[], v2freeze=[], not_v2reinitialize=[]):
        self.model = model
        self.when_far = when_far
        self.v2freeze = v2freeze
        self.not_v2reinitialize = not_v2reinitialize

    def far(self):
        # freeze
        print([w.name for w in self.model.weights])
        for w in self.model.trainable_variables:
            if w.name in self.v2freeze:
                w._trainable = False
        print([w.trainable for w in self.model.weights])

        # reinitialize
        variables_to_reset = [w.name for w in self.model.weights if w.name not in self.not_v2reinitialize]
        print(variables_to_reset)
        reset_weights(self.model, variables_to_reset)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.when_far:
            self.far()
