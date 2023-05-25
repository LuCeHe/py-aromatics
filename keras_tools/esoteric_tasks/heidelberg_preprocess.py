"""
Original version of this code was written by the authors of

Accurate and efficient time-domain classification
with adaptive spiking recurrent neural networks
Bojian Yin, Federico Corradi, and Sander M. Bohté

and can be found here

https://github.com/byin-cwi/Efficient-spiking-networks/blob/main/SHD/generate_dataset.py
"""

import os
import urllib.request
import gzip, shutil
# from keras.utils import get_file
import matplotlib.pyplot as plt

from GenericTools.stay_organized.download_utils import download_and_unzip

"""
The dataset is 48kHZ with 24bits precision
* 700 channels
* longest 1.17s
* shortest 0.316s
"""

# cache_dir=os.path.expanduser("~/data")
# cache_subdir="hdspikes"
# print("Using cache dir: %s"%cache_dir)
#
# # The remote directory with the data files
# base_url = "https://compneuro.net/datasets"

# Retrieve MD5 hashes from remote


# file_hashes = { line.split()[1]:line.split()[0] for line in lines if len(line.split())==2 }


FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
HEIDELBERGDIR = os.path.abspath(os.path.join(CDIR, '..', 'data', 'SpikingHeidelbergDigits'))
shd_train_filename = os.path.join(HEIDELBERGDIR, "shd_train.h5")

import tables
import numpy as np


def binary_image_readout(times, units, dt=1e-3):
    img = []
    N = int(1 / dt)
    for i in range(N):
        idxs = np.argwhere(times <= i * dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(700)
        vector[700 - vals] = 1
        times = np.delete(times, idxs)
        units = np.delete(units, idxs)
        img.append(vector)
    return np.array(img).astype(np.float16)


def binary_image_spatical(times, units, dt=1e-3, dc=10):
    img = []
    N = int(1 / dt)
    C = int(700 / dc)
    for i in range(N):
        idxs = np.argwhere(times <= i * dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(C)  # add spacial count
        vector[700 - vals] = 1
        times = np.delete(times, idxs)
        units = np.delete(units, idxs)
        img.append(vector)
    return np.array(img)


def generate_dataset(file_name, dt=1e-3):
    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    # This is how we access spikes and labels
    index = 0
    print("Number of samples: ", len(times))
    X = []
    y = []
    for i in range(len(times)):
        tmp = binary_image_readout(times[i], units[i], dt=dt)
        X.append(tmp)
        y.append(labels[i])
    return np.array(X), np.array(y)


def plot_heidelberg():
    k = 125
    fileh = tables.open_file(shd_train_filename, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    # This is how we access spikes and labels
    index = 0
    print("Times (ms):", times[index], max(times[index]))
    print("Unit IDs:", units[index])
    print("Label:", labels[index])

    plt.scatter(times[k], 700 - units[k], color="k", alpha=0.33, s=2)
    plt.title("Label %i" % labels[k])
    plt.xlabel('time [s]')
    plt.ylabel('channel')
    # plt.axis("off")
    plt.show()

# # how many time steps on each sample
# l = []
# for i in range(len(times)):
#     l.append(len(set(times[i])))
# print(max(l),np.argmax(l))
# print(min(l),np.argmin(l))
# plt.hist(l,bins=20)
# plt.show()
# # the sampling frequence of spoken digits
# l = []
# for i in range(len(times)):
#     a = np.array(sorted(list(set(times[i]))))
#     n = len(a)
#     l.append(min(a[1:]-a[:n-1]))
# print(max(l),np.argmax(l))
# print(min(l),np.argmin(l))
# plt.hist(l)
# plt.show()
#
# #  how many spoken digits longer than 1s
# l = []
# ll = []
# for i in range(len(times)):
#     l.append(max(times[i]))
#     if max(times[i])>1.: ll.append(i)
# print(max(l),np.argmax(l))
# print(min(l),np.argmin(l))
# plt.hist(l,bins=20)
# plt.show()
#
# def binary_image_readout(times,units,dt = 1e-3):
#     img = []
#     N = int(1/dt)
#     for i in range(N):
#         idxs = np.argwhere(times<=i*dt).flatten()
#         vals = units[idxs]
#         vals = vals[vals>0]
#         vector = np.zeros(700)
#         vector[700-vals] = 1
#         times = np.delete(times,idxs)
#         units = np.delete(units,idxs)
#         img.append(vector)
#     return np.array(img)
# idx = 1358
# tmp = binary_image_readout(times[idx],units[idx],dt=5e-3)
# plt.imshow(tmp.T)
# plt.show()
# # A quick raster plot for one of the samples
#
#
# fig = plt.figure(figsize=(16,4))
# idx = ll[:3]#[1979,1358,626]#np.random.randint(len(times),size=3)
# for i,k in enumerate(idx):
#     ax = plt.subplot(1,3,i+1)
#     ax.scatter(times[k],700-units[k], color="k", alpha=0.33, s=2)
#     ax.set_title("Label %i"%labels[k])
#     # ax.axis("off")
# plt.show()
#
# fig = plt.figure(figsize=(16,8))
# idx = ll[16:22]#[1979,1358,626]#np.random.randint(len(times),size=3)
# for i,k in enumerate(idx):
#     ax = plt.subplot(2,3,i+1)
#     ax.scatter(times[k],700-units[k], color="k", alpha=0.33, s=2)
#     ax.set_title("Label %i"%labels[k])
#     # ax.axis("off")
#
# plt.show()
