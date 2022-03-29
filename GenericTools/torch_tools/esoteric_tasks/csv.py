import os
import tensorflow as tf
import pandas as pd
import numpy as np

# FILENAME = os.path.realpath(__file__)
# CDIR = os.path.dirname(FILENAME)
# DATADIR = os.path.abspath(os.path.join(CDIR, '..', 'data'))
from GenericTools.torch_tools.esoteric_tasks.base_generator import BaseGenerator


class CsvTask(BaseGenerator):
    'Generates data for Keras'

    def __init__(
            self,
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            maxlen=3,
            data_split='train',
            in_features=['rsi', 'log_price_change', 'log_return_1_price', 'log_return_1_volume'],
            out_features=['10_price_up'],
            in_csv='path/file.csv',
            out_csv='path/file.csv',
    ):
        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            maxlen=maxlen,
        )


        in_csv = pd.read_csv(in_csv)
        dfs = [in_csv.filter(like=k, axis=1) for k in in_features]
        self.in_df = pd.concat(dfs, axis=1)

        out_csv = pd.read_csv(out_csv)
        dfs = [out_csv.filter(like=k, axis=1) for k in out_features]
        self.out_df = pd.concat(dfs, axis=1)

        del in_csv, dfs, out_csv

        self.n_samples = self.in_df.shape[0]
        self.on_epoch_end()

        self.epochs = 600 if epochs == None else epochs
        self.steps_per_epoch = int(self.n_samples / batch_size / maxlen) - 1 \
            if steps_per_epoch == None else min(steps_per_epoch, int(self.n_samples / batch_size / maxlen))

        self.in_dim = self.in_df.shape[1]
        self.out_dim = self.out_df.shape[1]

        self.out_columns = self.out_df.columns
        self.in_columns = self.in_df.columns

    def on_epoch_end(self):
        batch_beginnings = list(range(0, self.n_samples, self.maxlen))[:-1]
        np.random.shuffle(batch_beginnings)
        self.new_batch_beginnings = batch_beginnings

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        beginnings = self.new_batch_beginnings[-self.batch_size:]
        self.new_batch_beginnings = self.new_batch_beginnings[:-self.batch_size]

        input_batch = []
        output_batch = []

        for i in beginnings:
            past_split = self.in_df[i:i + self.maxlen].to_numpy()
            future_split = self.out_df[i:i + self.maxlen].to_numpy()

            input_batch.append(past_split[None])
            output_batch.append(future_split[None])

        input_batch = np.concatenate(input_batch, axis=0)
        output_batch = np.concatenate(output_batch, axis=0)

        return input_batch, output_batch


if __name__ == '__main__':
    pass