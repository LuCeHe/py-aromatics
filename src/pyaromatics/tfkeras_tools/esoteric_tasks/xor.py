import os
import numpy as np
import tensorflow as tf

from sg_design_lif.generate_data.base_generator import BaseGenerator

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'ptb'))
if not os.path.isdir(DATAPATH): os.mkdir(DATAPATH)

data_links = ['https://data.deepai.org/ptbdataset.zip']


class XorGenerator(BaseGenerator):
    'Generates data for Keras'

    def __init__(
            self,
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            maxlen=20,
            repetitions=2,
            train_val_test='train',
            data_path=DATAPATH,
            task_format=2):
        self.__dict__.update(epochs=epochs, steps_per_epoch=steps_per_epoch, batch_size=batch_size, maxlen=maxlen,
                             repetitions=repetitions, train_val_test=train_val_test, data_path=data_path,
                             task_format=task_format)

        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])
        self.z = np.concatenate([self.X, self.y, np.zeros_like(self.y)], 1)

        if self.task_format == 1:
            self.in_len = 3
            self.out_len = 3
            self.out_dim = 2

        elif self.task_format == 2:
            new_y = np.concatenate([2 * np.ones_like(self.X), self.y], axis=1)
            new_X = np.concatenate([self.X, 2 * np.ones_like(self.y)], axis=1)
            self.X = new_X
            self.y = new_y
            self.in_len = 3
            self.out_len = 3
            self.out_dim = 3

        else:
            raise NotImplementedError

        self.in_dim = 1

        self.epochs = 10 if epochs == None else epochs
        self.steps_per_epoch = 10 if steps_per_epoch == None else steps_per_epoch

    def data_generation(self):
        idx = np.random.randint(self.z.shape[0], size=self.batch_size)

        if self.task_format == 1:
            bx = self.z[idx][:, :-1]
            by = self.z[idx][:, 1:]
            by = tf.keras.utils.to_categorical(by, num_classes=self.out_dim)

            mask = np.zeros_like(bx)
            mask[:, 1] = 1

        elif self.task_format == 2:
            bx = self.X[idx]
            by = self.y[idx]
            by = tf.keras.utils.to_categorical(by, num_classes=self.out_dim)

            mask = np.zeros_like(bx)
            mask[:, -1] = 1

        else:
            raise NotImplementedError

        return {'input_spikes': bx[..., None], 'target_output': by, 'mask': mask[..., None]}


def test():
    batch_size = 3
    generator = XorGenerator(
        batch_size=batch_size,
        steps_per_epoch=1,
        maxlen=8, )

    batch = generator.data_generation()
    print(batch['input_spikes'].shape)
    print(batch['target_output'].shape)
    print(batch['target_output'].shape)
    print(batch['input_spikes'])
    print(batch['target_output'])


if __name__ == '__main__':
    test()
