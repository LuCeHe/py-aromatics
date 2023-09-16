import os
import matplotlib.pyplot as plt
import numpy as np

from pyaromatics.keras_tools.esoteric_tasks.long_range_arena.create_listops import listops_creation
from pyaromatics.keras_tools.esoteric_tasks.long_range_arena.listops import get_datasets
from pyaromatics.keras_tools.esoteric_tasks.base_generator import BaseGenerator

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATADIR = os.path.abspath(os.path.join(CDIR, '..', '..', '..', '..', 'data', 'lra', 'listops'))
os.makedirs(DATADIR, exist_ok=True)


class LRAGenerator(BaseGenerator):

    def __init__(
            self,
            task_name,
            epochs=1,
            tvt='train',
            batch_size=32,
            repetitions=1,
            steps_per_epoch=None,
            string_config='',
    ):
        assert task_name in ['listops']

        if len(os.listdir(DATADIR)) == 0 and task_name == 'listops':
            listops_creation()

        if task_name == 'listops':
            length = 2000
            classes = 10

        self.__dict__.update(batch_size=batch_size, tvt=tvt,
                             steps_per_epoch=steps_per_epoch, repetitions=repetitions)

        self.on_epoch_end()

        self.in_dim = 1
        self.out_dim = classes

        self.in_len = length * repetitions
        self.out_len = length * repetitions
        self.epochs = 100 if epochs == None else epochs

        self.steps_per_epoch = int(self.n_samples / self.batch_size) \
            if steps_per_epoch == None else steps_per_epoch

    def on_epoch_end(self):
        datasets = get_datasets(8, 'listops', batch_size=self.batch_size)

        self.vocab_size = datasets['vocab_size']
        if self.tvt == 'test':
            self.n_samples = datasets['n_test_samples']
            ds = datasets['test']

        elif self.tvt in ['validation', 'val']:
            self.n_samples = datasets['n_val_samples']
            ds = datasets['val']

        elif self.tvt == 'train':
            self.n_samples = datasets['n_train_samples']
            ds = datasets['train']

        else:
            raise NotImplementedError

        self.iterds = iter(ds)

    def data_generation(self):
        batch = next(self.iterds)

        input = batch['inputs']
        target = batch['targets']
        target = np.repeat(target[..., None], self.out_len, 1)
        return {'input_spikes': input, 'target_output': target, 'mask': 1.}


def test_generator():
    gen = LRAGenerator(
        task_name='listops',
        epochs=1,
        tvt='train',
        batch_size=32,
        steps_per_epoch=1,
    )
    print(gen.steps_per_epoch)
    for i in range(gen.steps_per_epoch):
        batch = gen.__getitem__()
        print(i, [b.shape for b in batch[0]])

    image = batch[0][0]
    print(image.shape)
    fig, ax = plt.subplots(1, 1, figsize=(6, 15), gridspec_kw={'hspace': 0})
    im = ax.pcolormesh(image[0])
    plt.show()


if __name__ == '__main__':
    test_generator()
