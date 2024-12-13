'''
original: https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
'''
from tensorflow.keras.callbacks import Callback
import warnings
import numpy as np


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0, mode='auto'):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if self.monitor_op(current, self.value):
            if self.verbose > 0:
                print("Epoch {}: early stopping".format(epoch))
            self.model.stop_training = True
