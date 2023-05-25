import numpy as np
import tensorflow as tf


def create_sequential_mnist(path='mnist.npz', batch_size=20, n_input=100, cue_duration=20, is_test=False,
                            worker_id=0, n_workers=1):
    def map_fn(_x, _y):
        thresholds = tf.cast(tf.linspace(0., 254., (n_input - 1) // 2), tf.uint8)
        _x = tf.reshape(_x, (-1,))
        lower = _x[:, None] < thresholds[None, :]
        higher = _x[:, None] >= thresholds[None, :]
        transition_onset = tf.logical_and(lower[:-1], higher[1:])
        transition_offset = tf.logical_and(higher[:-1], lower[1:])
        onset_spikes = tf.cast(transition_onset, tf.float32)
        onset_spikes = tf.concat((onset_spikes, tf.zeros_like(onset_spikes[:1])), 0)
        offset_spikes = tf.cast(transition_offset, tf.float32)
        offset_spikes = tf.concat((offset_spikes, tf.zeros_like(offset_spikes[:1])), 0)

        touch_spikes = tf.cast(tf.equal(_x, 255), tf.float32)[..., None]

        out_spikes = tf.concat((onset_spikes, offset_spikes, touch_spikes), -1)
        out_spikes = tf.tile(out_spikes[:, None], (1, 2, 1))
        out_spikes = tf.reshape(out_spikes, (-1, n_input - 1))
        out_spikes = tf.concat((out_spikes, tf.zeros_like(out_spikes[:cue_duration])), 0)
        signal_spikes = tf.concat(
            (tf.zeros_like(out_spikes[:-cue_duration, :1]), tf.ones_like(touch_spikes[:cue_duration])), 0)
        out_spikes = tf.concat((out_spikes, signal_spikes), -1)

        return out_spikes, _y

    train_data, test_data = tf.keras.datasets.mnist.load_data(path=path)
    data = (train_data, test_data)

    if is_test:
        d = test_data
    else:
        d = train_data
    x, y = d

    samples_per_worker = int(x.shape[0] / n_workers)
    x = x[worker_id * samples_per_worker:(worker_id + 1) * samples_per_worker]
    y = y[worker_id * samples_per_worker:(worker_id + 1) * samples_per_worker]
    y = y.astype(np.int32)
    data_set = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(samples_per_worker).map(map_fn).batch(batch_size)
    return data_set

