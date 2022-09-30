import tensorflow as tf
import numpy as np


class NumpyClassificationGenerator(tf.keras.utils.Sequence):

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __init__(
            self,
            X, y,
            epochs=1,
            batch_size=32,
            steps_per_epoch=None,
            output_type='[i]o'
    ):

        self.__dict__.update(batch_size=batch_size, output_type=output_type, X=X, y=y,
                             steps_per_epoch=steps_per_epoch)

        self.on_epoch_end()
        self.in_dims = X.shape
        self.out_dims = (*y.shape, np.max(y)+1)
        self.epochs = 450 if epochs == None else epochs

        self.batch_size = batch_size
        self.batch_index = 0

        self.steps_per_epoch = int(np.floor((self.X.shape[0]) / self.batch_size)) \
            if steps_per_epoch == None or steps_per_epoch<0 else steps_per_epoch

    def on_epoch_end(self):
        self.batch_index = 0
        self.random_indices = np.array(list(range(self.X.shape[0])))
        np.random.shuffle(self.random_indices)

    def data_generation(self):
        indices = self.random_indices[:self.batch_size]
        self.random_indices = self.random_indices[self.batch_size:]
        batch = self.X[indices]
        target = self.y[indices]
        # target = np.repeat(target[..., None], self.length, 1)
        return {'input_tensor': batch, 'output_tensor': target, 'mask': 1.}

    def __getitem__(self, index=0):
        batch = self.data_generation()
        i, m, o = batch['input_tensor'], batch['mask'], batch['output_tensor']

        if self.output_type == '[im]o':
            return (i, m), o
        elif self.output_type == '[imo]':
            return (i, m, o),
        if self.output_type == '[io]':
            return (i, o),
        if self.output_type == '[i]o':
            return i, o
        else:
            raise NotImplementedError
