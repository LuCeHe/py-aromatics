from tensorflow.keras.datasets import cifar10
import tensorflow.keras as keras

def getCifar(steps_per_epoch=None):
    num_classes = 10
    img_rows, img_cols = 32, 32

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print('Train: X = %s, y = %s' % (x_train.shape, y_train.shape))
    print('Test:  X = %s, y = %s' % (x_test.shape, y_test.shape))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if not steps_per_epoch == None:
        x_train, y_train, x_test, y_test = x_train[:64], \
                                           y_train[:64], \
                                           x_test[:64], \
                                           y_test[:64]

    return x_train, y_train, x_test, y_test

