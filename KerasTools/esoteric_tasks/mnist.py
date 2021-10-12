import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np


def getMNIST(categorical=True, sequential=False, original_size=True,
             training_set='all', train_split=.8, normalize=True, spike_latency=False):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if training_set == 'test':
        x, y = x_test, y_test
    elif training_set == 'train':
        n_samples_train = x_train.shape[0]
        x = x_train[:int(train_split * n_samples_train)]
        y = y_train[:int(train_split * n_samples_train)]
    elif training_set in ['validation', 'val']:
        n_samples_train = x_train.shape[0]
        x = x_train[int(train_split * n_samples_train):]
        y = y_train[int(train_split * n_samples_train):]
    else:
        raise NotImplementedError

    num_classes = max(y_train) + 1

    if not original_size:
        x = x[:, ::3, ::3]

    n_samples = x.shape[0]

    img_rows, img_cols = x.shape[1], x.shape[2]

    if K.image_data_format() == 'channels_first':
        x = x.reshape(n_samples, 1, img_rows, img_cols)
    else:
        x = x.reshape(n_samples, img_rows, img_cols, 1)

    if sequential:
        x = x.reshape((-1, img_rows * img_cols))

    if normalize:
        x = x.astype('float32')
        x /= 255

    if spike_latency:
        # original: The Remarkable Robustness of Surrogate Gradient
        # Learning for Instilling Complex Function in Spiking
        # Neural Networks
        tau_eff = 50.
        n_dt_per_step = 10
        thr = .2
        t_inf = -1
        x = x * tau_eff

        T = t_inf * np.ones_like(x)

        idx = x > thr
        T[idx] = tau_eff * np.log(x[idx] / (x[idx] - thr)) * n_dt_per_step

        # T = tau_eff * np.log(x / (x - thr)) * (x > thr) + t_inf * (x <= thr)
        x = T.round(0)

    # convert class vectors to binary class matrices
    if categorical:
        y = tf.keras.utils.to_categorical(y_train, num_classes)

    return x, y


if __name__ == '__main__':
    x, y = getMNIST(categorical=False, sequential=True, original_size=True, training_set='train', train_split=.8,
                    normalize=True)

    tau_eff = 50.
    n_dt_per_step = 10
    thr = .2
    t_inf = -1
    x = x * tau_eff

    T = t_inf * np.ones_like(x)

    idx = x > thr
    T[idx] = tau_eff * np.log(x[idx] / (x[idx] - thr)) * n_dt_per_step

    # T = tau_eff * np.log(x / (x - thr)) * (x > thr) + t_inf * (x <= thr)
    T = T.round(0)
    print(T[1])
    # spike_latency

    import matplotlib.pyplot as plt

    t = T.flatten().tolist()
    t = list(filter((t_inf).__ne__, t))

    n, bins, patches = plt.hist(x=t, bins=50, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.show()

