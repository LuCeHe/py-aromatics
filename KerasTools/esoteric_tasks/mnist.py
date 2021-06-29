import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K


def getMNIST(categorical=True, sequential=False, original_size=True,
             training_set='all', train_split=.8, normalize=True):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_classes = max(y_train) + 1

    if not original_size:
        x_train, x_test = x_train[:, ::3, ::3], x_test[:, ::3, ::3]

    n_samples_train = x_train.shape[0]
    n_samples_test = x_test.shape[0]

    img_rows, img_cols = x_train.shape[1], x_train.shape[2]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(n_samples_train, 1, img_rows, img_cols)
        x_test = x_test.reshape(n_samples_test, 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(n_samples_train, img_rows, img_cols, 1)
        x_test = x_test.reshape(n_samples_test, img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    if sequential:
        x_train = x_train.reshape((-1, img_rows * img_cols))
        x_test = x_test.reshape((-1, img_rows * img_cols))

    if normalize:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

    # convert class vectors to binary class matrices
    if categorical:
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    if training_set == 'all':
        return x_train, y_train, x_test, y_test
    elif training_set == 'test':
        return x_test, y_test
    elif training_set == 'train':
        x = x_train[:int(train_split * n_samples_train)]
        y = y_train[:int(train_split * n_samples_train)]
        return x, y
    elif training_set in ['validation', 'val']:
        x = x_train[int(train_split * n_samples_train):]
        y = y_train[int(train_split * n_samples_train):]
        return x, y
    else:
        raise NotImplementedError
