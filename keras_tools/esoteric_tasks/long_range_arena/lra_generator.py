import os, shutil
import matplotlib.pyplot as plt
import numpy as np

from pyaromatics.keras_tools.esoteric_tasks.long_range_arena.create_listops import listops_creation
from pyaromatics.keras_tools.esoteric_tasks.long_range_arena.images import get_cifar10_datasets, \
    get_pathfinder_orig_datasets, get_pathfinder_base_datasets
from pyaromatics.keras_tools.esoteric_tasks.long_range_arena.listops import get_listops_datasets
from pyaromatics.keras_tools.esoteric_tasks.base_generator import BaseGenerator
from pyaromatics.keras_tools.esoteric_tasks.long_range_arena.retrieval import get_matching_datasets
from pyaromatics.keras_tools.esoteric_tasks.long_range_arena.text_classification import get_tc_datasets
from pyaromatics.stay_organized.download_utils import download_and_unzip

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATADIR = os.path.abspath(os.path.join(CDIR, '..', '..', '..', '..', 'data', 'lra'))
LODIR = os.path.join(DATADIR, 'listops')
PFDIR = os.path.join(DATADIR, 'pathfinder')
RTDIR = os.path.join(DATADIR, 'retrieval')
EXTRA = os.path.join(DATADIR, 'lra_release')
RTDIR_tmp = os.path.join(EXTRA, r'lra_release\tsv_data')
for d in [DATADIR, LODIR, RTDIR]:
    os.makedirs(d, exist_ok=True)


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
        assert task_name in ['listops', 'scifar', 'pathfinder', 'pathx', 'text', 'retrieval']

        if len(os.listdir(LODIR)) == 0 and task_name == 'listops':
            listops_creation()

        if len(os.listdir(PFDIR)) == 0 and (task_name == 'pathfinder' or task_name == 'pathx'):
            url = 'https://storage.cloud.google.com/long-range-arena/pathfinder_tfds.gz'
            download_and_unzip([url], PFDIR)

        if len(os.listdir(RTDIR)) == 0 and task_name == 'retrieval':
            url = 'https://storage.googleapis.com/long-range-arena/lra_release.gz'
            download_and_unzip([url], RTDIR, unzip_what='new_aan_pairs')
            for d in os.listdir(RTDIR_tmp):
                os.rename(os.path.join(RTDIR_tmp, d), os.path.join(RTDIR, d))

        if task_name == 'listops':
            length = 2000
            classes = 10
            self.get_datasets = get_listops_datasets

        elif task_name == 'scifar':
            length = 32 * 32
            classes = 10
            self.get_datasets = get_cifar10_datasets

        elif task_name == 'pathfinder':
            length = 32 * 32
            classes = 2
            self.get_datasets = lambda batch_size: get_pathfinder_base_datasets(
                batch_size=batch_size,
                resolution=32,
                split='hard'
            )

        elif task_name == 'pathx':
            length = 128 * 128
            classes = 2
            self.get_datasets = lambda batch_size: get_pathfinder_base_datasets(
                batch_size=batch_size,
                resolution=128,
                split='hard'
            )

        elif task_name == 'text':
            length = 4000
            classes = 2
            self.get_datasets = lambda batch_size: get_tc_datasets(
                'imdb_reviews',
                batch_size=batch_size,
                fixed_vocab=None,
                max_length=length,
                tokenizer='char'
            )

        elif task_name == 'retrieval':
            for d in os.listdir(RTDIR_tmp):
                os.rename(os.path.join(RTDIR_tmp, d), os.path.join(RTDIR, d))
            shutil.rmtree(EXTRA)
            length = 8000
            classes = 2
            self.get_datasets = lambda batch_size: get_matching_datasets(
                batch_size=batch_size,
                data_dir=RTDIR,
                fixed_vocab=None,
                max_length=length//2,
                tokenizer='char'
            )

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

        print('vocab size', self.vocab_size)

    def on_epoch_end(self):
        datasets = self.get_datasets(batch_size=self.batch_size)

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default='retrieval',
        type=str,
        help="url from where to download",
    )
    args = parser.parse_args()

    gen = LRAGenerator(
        task_name=args.task_name,
        epochs=1,
        tvt='train',
        batch_size=3,
        steps_per_epoch=1,
    )
    print(gen.steps_per_epoch)
    for i in range(gen.steps_per_epoch):
        batch = gen.__getitem__()
        print(i, [b.shape for b in batch[0]])

    images = batch[0][0]
    classes = batch[0][1]
    print(images[0].tolist())
    print(classes[0])


if __name__ == '__main__':
    test_generator()

    url = 'https://storage.cloud.google.com/long-range-arena/lra_release/lra_release/tsv_data.gz'