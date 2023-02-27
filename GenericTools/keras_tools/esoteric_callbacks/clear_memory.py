import psutil
import GPUtil
import gc
import tensorflow as tf
from tensorflow import keras as k



class ClearMemory(tf.keras.callbacks.Callback):
    def __init__(self, end_of_epoch=True, end_of_batch=True, batch_frequency=1000, verbose=1):
        super().__init__()
        self.end_of_epoch = end_of_epoch
        self.end_of_batch = end_of_batch
        self.batch_frequency = batch_frequency
        self.verbose = verbose

    def clear_memory(self):
        if self.verbose > 0:
            print(f'\n\n')
            print('-' * 30)
            print('Clearing memory')
            print('before')
            bcpup = psutil.cpu_percent(4)
            bramp = psutil.virtual_memory().percent
            bram = f"{psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB"
            print('GPU utilization')
            GPUtil.showUtilization(all=True)

        gc.collect()
        tf.compat.v1.keras.backend.clear_session()

        gc.collect()
        tf.compat.v1.keras.backend.clear_session()

        if self.verbose > 0:
            acpup = psutil.cpu_percent(4)
            aramp = psutil.virtual_memory().percent
            aram = f"{psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB"


            print('\nafter')
            print('GPU utilization')
            GPUtil.showUtilization(all=True)

            print(f'CPU usage %:  {bcpup}% -> {acpup}%')
            print(f'RAM memory %: {bramp}% -> {aramp}%')
            print(f'RAM used:     {bram} -> {aram}')


            print('-' * 30)
            print(f'\n\n')

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.batch_frequency == 0 and self.end_of_batch:
            self.clear_memory()

    def on_epoch_end(self, epoch, logs=None):
        if self.end_of_epoch:
            self.clear_memory()
