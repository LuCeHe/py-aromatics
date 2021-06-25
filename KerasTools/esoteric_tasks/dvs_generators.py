import ebdataset, os
import numpy as np
import tensorflow as tf
from GenericTools.StayOrganizedTools.download_utils import download_url
from ebdataset.vision import IBMGesture, H5IBMGesture
from quantities import ms

np.set_printoptions(precision=2, threshold=20)

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATAFOLDER = os.path.abspath(os.path.join(CDIR, '..\data\DVSGesture'))
h5_path = os.path.abspath(os.path.join(CDIR, '..\data\DVSGesture.h5'))
# With sparse representation:
for spike_train, label in IBMGesture(DATAFOLDER).train_values_generator():
    print(spike_train.x)
    print(spike_train.y)
    print(spike_train.p)
    print(spike_train.ts)
    print(max(spike_train.ts), max(spike_train.x), max(spike_train.y))
    break

if not os.path.isfile(h5_path):
    H5IBMGesture.convert(DATAFOLDER, h5_path)
generator = H5IBMGesture(h5_path)

spike_train, label = generator.__getitem__(0)

print(spike_train.shape, label.shape)
print(label)

indices = list(map(list, zip(spike_train.ts, spike_train.x, spike_train.y)))
print(max(indices))
sparse = tf.sparse.SparseTensor(indices=indices, values=spike_train.p * 2 - 1,
                                dense_shape=[max(spike_train.ts), max(spike_train.x), max(spike_train.y)])
# dense = tf.sparse.to_dense(sparse)

max_t = 6e6
in_dim = 128
in_len = int(max_t / 1000)
times = spike_train.ts
units_w, units_h = spike_train.x, spike_train.y
values = spike_train.p * 2 - 1
i = 0

bins = np.linspace(0, max_t, in_len)
dense = np.zeros((in_dim, in_dim, in_len))
tr = times
which = tr < max_t
tr = tr[which]
v = values[which]
u_w = in_dim - 1 - units_w[which]
u_h = in_dim - 1 - units_h[which]
binned = np.digitize(tr, bins, right=True)  # times[k]
print('in_len:  ', in_len)
print('times:   ', times)
print('bins:    ', bins)
print('binned:  ', binned)
print('u_w:     ', u_w)
print('max u_w: ', max(u_w))
print('max_x:   ', max(spike_train.x), min(spike_train.x), np.median(spike_train.x))
outliers = spike_train.x>128
print('outliers {}/{}'.format(sum(outliers), len(outliers)))

dense[u_w, u_h, binned] = v

print(dense.shape)

class DVSGestureAsLanguage(tf.keras.utils.Sequence):
    # PROs: use embeddings
    # CONs: very long sentences
    pass


class DVSGestureAsImages(tf.keras.utils.Sequence):
    # PROs:
    # CONs:
    pass
