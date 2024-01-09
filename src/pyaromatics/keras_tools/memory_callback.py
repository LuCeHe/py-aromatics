import os
import tensorflow as tf
import psutil
import py3nvml.py3nvml as nvidia_smi

import pandas as pd


class TrackMemory(tf.keras.callbacks.Callback):
    def __init__(self, filepath, print_every=1, ):
        self.print_every = print_every
        self.filepath = filepath

        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    def save_memory_usage(self, epoch):
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        # TODO: this reads the gpu after the batch is read, not during, which might be of higher interest
        dictionary_memory = {}
        dictionary_memory.update({'GPU_Gb': mem_res.used / (1024 ** 2),
                                  'GPU_percentage': 100 * (mem_res.used / mem_res.total)})
        dictionary_memory.update(
            {'CPU_{}'.format(i): m for i, m in enumerate(psutil.cpu_percent(interval=1, percpu=True))})
        dictionary_memory.update({'virtual_memory_percent': psutil.virtual_memory().percent})
        dictionary_memory.update({'swap_memory_percent': psutil.swap_memory().percent})

        # if not os.name == 'nt':
        #     dictionary_memory.update(psutil.sensors_temperatures())
        user_names = '---'.join([u.name.replace(' ', '_') for u in psutil.users()])
        dictionary_memory.update({'user_names': user_names})

        df = pd.DataFrame(dictionary_memory, index=[epoch])
        if not os.path.isfile(self.filepath):
            df.to_csv(self.filepath, mode='a', header=True)
        else:
            df.to_csv(self.filepath, mode='a', header=False)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self.save_memory_usage(-1)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_every == 0:
            self.save_memory_usage(epoch)
