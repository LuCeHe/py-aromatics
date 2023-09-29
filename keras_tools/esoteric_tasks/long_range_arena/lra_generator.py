import os, shutil, json
import argparse

import numpy as np

from pyaromatics.keras_tools.esoteric_tasks import lra_tasks
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
EXTRA = os.path.join(RTDIR, 'lra_release')
RTDIR_tmp = os.path.join(EXTRA, 'lra_release', 'tsv_data')
for d in [DATADIR, LODIR, RTDIR]:
    os.makedirs(d, exist_ok=True)

meta_data_path = os.path.join(DATADIR, 'meta_data.json')
# if it doesn't exist, create empty meta_data with json
if not os.path.exists(meta_data_path):
    meta_data = {task: {'vocab_size': 0, 'n_train_samples': 0, 'n_val_samples': 0, 'n_test_samples': 0}
                 for task in lra_tasks}
    # save with json not with pandas
    with open(meta_data_path, 'w') as f:
        json.dump(meta_data, f)
    del meta_data






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
        assert task_name in lra_tasks

        if len(os.listdir(LODIR)) == 0 and task_name == 'listops':
            listops_creation()

        if len(os.listdir(PFDIR)) == 0 and (task_name == 'pathfinder' or task_name == 'pathx'):
            url = 'https://storage.cloud.google.com/long-range-arena/pathfinder_tfds.gz'
            download_and_unzip([url], PFDIR)

        if len(os.listdir(RTDIR)) == 0 and task_name == 'retrieval':
            url = 'https://storage.googleapis.com/long-range-arena/lra_release.gz'
            download_and_unzip([url], RTDIR, unzip_what='new_aan_pairs')

            if os.path.exists(RTDIR_tmp):
                for d in os.listdir(RTDIR_tmp):
                    os.rename(os.path.join(RTDIR_tmp, d), os.path.join(RTDIR, d))
            shutil.rmtree(EXTRA)

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
            length = 8000
            classes = 2
            self.get_datasets = lambda batch_size: get_matching_datasets(
                batch_size=batch_size,
                data_dir=RTDIR,
                fixed_vocab=None,
                max_length=length // 2,
                tokenizer='char'
            )

        self.__dict__.update(batch_size=batch_size, tvt=tvt,
                             steps_per_epoch=steps_per_epoch, repetitions=repetitions)

        self.in_dim = 1
        self.out_dim = classes

        self.in_len = length * repetitions
        self.out_len = length * repetitions
        self.epochs = 100 if epochs == None else epochs

        # load json meta_data
        with open(meta_data_path, 'r') as f:
            meta_data = json.load(f)

        # get task column
        task_col = meta_data[task_name]
        # get number of samples
        self.n_samples = task_col[f'n_{tvt}_samples']
        # get vocab size
        self.vocab_size = task_col['vocab_size']

        if self.n_samples == 0:
            self.on_epoch_begin()
            self.n_samples = self.n_samples
            self.vocab_size = self.vocab_size
            self.on_epoch_end()
            # save it in the csv
            meta_data[task_name][f'n_{tvt}_samples'] = self.n_samples
            meta_data[task_name]['vocab_size'] = self.vocab_size

            # save with json not with pandas
            with open(meta_data_path, 'w') as f:
                json.dump(meta_data, f)

        del meta_data

        self.steps_per_epoch = int(self.n_samples / self.batch_size) - 1 \
            if steps_per_epoch == None else steps_per_epoch

    def on_epoch_begin(self):
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

    def on_epoch_end(self):
        del self.iterds

    def data_generation(self):
        if not hasattr(self, 'iterds'):
            self.on_epoch_begin()

        batch = next(self.iterds)

        input = batch['inputs'][..., None]
        target = batch['targets']
        target = np.repeat(target[..., None], self.out_len, 1)
        # print('inside generator:', input.shape, target.shape)
        return {'input_spikes': input, 'target_output': target, 'mask': 1.}


def test_generator():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default='listops',
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
    print('vocab_size:', gen.vocab_size)
    print('samples:', gen.n_samples)
    print(gen.steps_per_epoch)
    gen.on_epoch_begin()
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
