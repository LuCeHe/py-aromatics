# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Callback that stops training when a specified amount of time has passed."""
"""
original in tensorflow_addons, we add the possibility of stopping within an epoch
"""

import datetime
import time
from typeguard import typechecked

import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class TimeStopping(Callback):
    """Stop training when a specified amount of time has passed.

    Args:
        seconds: maximum amount of time before stopping.
            Defaults to 86400 (1 day).
        verbose: verbosity mode. Defaults to 0.
    """

    @typechecked
    def __init__(self, seconds: int = 86400, verbose: int = 0, stop_within_epoch=False):
        super().__init__()

        self.seconds = seconds
        self.verbose = verbose
        self.stopped_epoch = None
        self.epochs = 0
        self.stop_within_epoch = stop_within_epoch

    def on_train_begin(self, logs=None):
        self.starting_time = time.time()
        self.stopping_time = self.starting_time + self.seconds

    def on_train_batch_end(self, batch, logs=None):
        if self.stop_within_epoch and time.time() >= self.stopping_time:
            self.model.stop_training = True
            self.stopped_epoch = self.epochs
    def on_epoch_end(self, epoch, logs={}):
        self.epochs += 1
        extra_epoch_time = (time.time() - self.starting_time) / self.epochs
        if time.time() + extra_epoch_time >= self.stopping_time:
            self.model.stop_training = True
            self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None and self.verbose > 0:
            formatted_time = datetime.timedelta(seconds=self.seconds)
            msg = "Timed stopping at epoch {} after training for {}".format(
                self.stopped_epoch + 1, formatted_time
            )
            print(msg)

    def get_config(self):
        config = {
            "seconds": self.seconds,
            "verbose": self.verbose,
        }

        base_config = super().get_config()
        return {**base_config, **config}


class PreTimeStopping(Callback):
    """Stop training when a specified amount of time has passed.

    Args:
        seconds: maximum amount of time before stopping.
            Defaults to 86400 (1 day).
        verbose: verbosity mode. Defaults to 0.
    """

    @typechecked
    def __init__(self, seconds: int = 86400, verbose: int = 0, stop_where='epoch'):
        super().__init__()

        self.seconds = seconds
        self.verbose = verbose
        self.stopped_epoch = None
        self.epoch = 0
        self.where = stop_where

        if stop_where == 'epoch':
            # TODO: estimate if next epoch is going to go beyond, and stop preemptively
            self.on_epoch_end = self.on_x_end
        elif stop_where == 'batch':
            # FIXME: doesn't work yet
            self.on_batch_end = self.on_x_end
        else:
            raise NotImplementedError

    def on_train_begin(self, logs=None):
        self.stopping_time = time.time() + self.seconds

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_x_end(self, epoch, logs={}):
        print(epoch)
        print(time.time())
        print(self.stopping_time)
        print(self.model.stop_training)
        if time.time() >= self.stopping_time:
            self.model.stop_training = True
            self.stopped_epoch = self.epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None and self.verbose > 0:
            formatted_time = datetime.timedelta(seconds=self.seconds)
            msg = "Timed stopping at epoch {} after training for {}".format(
                self.stopped_epoch + 1, formatted_time
            )
            print(msg)

    def get_config(self):
        config = {
            "seconds": self.seconds,
            "verbose": self.verbose,
            "stop_where": self.stop_where
        }

        base_config = super().get_config()
        return {**base_config, **config}
