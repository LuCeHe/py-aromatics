import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
import math
from sklearn.model_selection import train_test_split


class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    """
    source:
    https://towardsdatascience.com/keras-custom-data-generators-example-with-mnist-dataset-2a7a2d2b0360
    """

    def __init__(self, X_data, y_data, batch_size, dim, n_classes,
                 to_fit, shuffle=True):
        self.batch_size = batch_size
        self.X_data = X_data
        self.labels = y_data
        self.y_data = y_data
        self.to_fit = to_fit
        self.n_classes = n_classes
        self.dim = dim
        self.shuffle = shuffle
        self.n = 0
        self.list_IDs = np.arange(len(self.X_data))
        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0

        return data

    def __len__(self):
        # Return the number of batches of the dataset
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, index=None):
        if index == None:
            index = np.random.choice(len(self.indexes))

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:
                               (index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self._generate_x(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.X_data))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_x(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.dim))

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self.X_data[ID]

            # Normalize data
            X = (X / 255).astype('float32')

        return X[:, :, :, np.newaxis]

    def _generate_y(self, list_IDs_temp):

        y = np.empty(self.batch_size)

        for i, ID in enumerate(list_IDs_temp):
            y[i] = self.y_data[ID]

        return keras.utils.to_categorical(
            y, num_classes=self.n_classes)


def mnist_preprocessed(sequential=False, generator=False, batch_size=64):
    num_classes = 10
    img_rows, img_cols = 28, 28

    config = lambda x: x
    config.num_classes = num_classes
    config.img_rows = img_rows
    config.img_cols = img_cols
    config.channels = 1
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if sequential:
        x_train = x_train.reshape((-1, img_rows * img_cols))
        x_test = x_test.reshape((-1, img_rows * img_cols))

    if generator:
        input_shape = (img_rows, img_cols)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

        train_generator = DataGenerator(x_train, y_train, batch_size=batch_size,
                                        dim=input_shape, n_classes=num_classes,
                                        to_fit=True, shuffle=True)
        val_generator = DataGenerator(x_val, y_val, batch_size=batch_size,
                                      dim=input_shape, n_classes=num_classes,
                                      to_fit=True, shuffle=True)
        test_generator = DataGenerator(x_test, y_test, batch_size=batch_size,
                                       dim=input_shape, n_classes=num_classes,
                                       to_fit=True, shuffle=True)

        return train_generator, val_generator, test_generator, config

    else:

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

        return x_train, x_val, x_test, y_train, y_val, y_test
