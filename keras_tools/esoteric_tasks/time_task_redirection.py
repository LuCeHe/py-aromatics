import os

# from sg_design_lif.generate_data.bit_generator import OneBitTimeDependentGenerator
# from sg_design_lif.generate_data.grammar_generators import CFG_AutoEncoding_Generator, MergeSearch
from pyaromatics.keras_tools.esoteric_tasks.heidelberg_generator import SpokenHeidelbergDigits
from pyaromatics.keras_tools.esoteric_tasks.long_range_arena.lra_generator import LRAGenerator, lra_tasks
# from sg_design_lif.generate_data.huggingface_generator import HuggingfaceGenerator
# from sg_design_lif.generate_data.lca_generator import LCAGenerator
from pyaromatics.keras_tools.esoteric_tasks.mnist_generators import SeqMNIST
# from sg_design_lif.generate_data.monkey_generator import MonkeyGenerator
from pyaromatics.keras_tools.esoteric_tasks.ptb_generator import PTBGenerator

# from sg_design_lif.generate_data.random_generator import RandomGenerator
# from sg_design_lif.generate_data.xor import XorGenerator
from pyaromatics.keras_tools.esoteric_tasks.random_generator import RandomGenerator

# from pyaromatics.keras_tools.esoteric_tasks.grammar_generators import CFG_AutoEncoding_Generator

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATADIR = os.path.abspath(os.path.join(CDIR, '..', '..', '..', 'data'))
os.makedirs(DATADIR, exist_ok=True)
STATSPATH = os.path.join(DATADIR, 'task_stats.csv')

language_tasks = ['ptb', 'wiki103', 'wmt14', 'time_ae_merge', 'monkey', 'wordptb', 'wordptb1', ] + \
                 ['lra_' + t for t in lra_tasks]


def Task(timerepeat=1, batch_size=64, steps_per_epoch=None, epochs=1, name='time_ae', train_val_test='train',
         neutral_phase_length=0, category_coding='onehot', inherit_from_gen=False, maxlen=100, output_type='[io]',
         comments=''):
    if name == 'time_ae':
        gen = CFG_AutoEncoding_Generator(
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            maxlen=maxlen)

    elif name == 'onebit':
        gen = OneBitTimeDependentGenerator(
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            timesteps_dependency=3,
            maxlen=maxlen,
            neutral_phases=True,
            repetitions=timerepeat)

    elif 'ptb' == name:
        gen = PTBGenerator(
            data_dir=DATADIR,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen,
            repetitions=timerepeat,
            train_val_test=train_val_test,
            neutral_phase_length=neutral_phase_length,
            category_coding='',
            config=comments)

    elif 'wordptb' == name:
        gen = PTBGenerator(
            data_dir=DATADIR,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen,
            repetitions=timerepeat,
            train_val_test=train_val_test,
            neutral_phase_length=neutral_phase_length,
            category_coding='',
            char_or_word='word',
            config=comments)


    elif 'wordptb1' == name:
        gen = PTBGenerator(
            data_dir=DATADIR,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen,
            repetitions=timerepeat,
            train_val_test=train_val_test,
            neutral_phase_length=neutral_phase_length,
            category_coding='',
            char_or_word='word',
            config=comments + '_ptb1')

    elif 'wordptb_oh' == name:
        gen = PTBGenerator(
            data_dir=DATADIR,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen,
            repetitions=timerepeat,
            train_val_test=train_val_test,
            neutral_phase_length=neutral_phase_length,
            category_coding='onehot',
            char_or_word='word')

    elif name.startswith('lra_'):
        task_name = name.replace('lra_', '')
        gen = LRAGenerator(
            task_name,
            epochs=epochs,
            tvt=train_val_test,
            batch_size=batch_size,
            repetitions=timerepeat,
            steps_per_epoch=steps_per_epoch,
            string_config='',
        )

    elif 'wiki103' == name:
        gen = Wiki103Generator(
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen,
            repetitions=timerepeat,
            train_val_test=train_val_test,
            neutral_phase_length=neutral_phase_length)

    elif name == 'simplest_random':
        gen = RandomGenerator(
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen)

    elif name == 'time_ae_merge':
        gen = MergeSearch(
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            n_hot=1,
            time_period=timerepeat,
            maxlen=maxlen)

    elif name == 's_mnist':
        gen = SeqMNIST(
            data_dir=DATADIR,
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            permuted=False)

    elif name == 'ps_mnist':
        gen = SeqMNIST(
            data_dir=DATADIR,
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            permuted=True)

    elif name == 'sl_mnist':
        gen = SeqMNIST(
            data_dir=DATADIR,
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            permuted=False,
            spike_latency=True)

    elif name == 'small_s_mnist':
        gen = SeqMNIST(
            data_dir=DATADIR,
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            original_size=False
        )

    elif name == 'heidelberg':
        gen = SpokenHeidelbergDigits(
            data_dir=DATADIR,
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            string_config=comments
        )


    elif name == 'lca':
        gen = LCAGenerator(
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            config=comments
        )

    elif name == 'wmt14':
        gen = Wmt14Generator(
            epochs=epochs,
            batch_size=batch_size,
            train_val_test=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat,
            category_coding=category_coding)

    elif name == 'monkey':
        gen = MonkeyGenerator(
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch)

    elif name == 'xor':
        gen = XorGenerator(
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            repetitions=timerepeat)

    elif 'huggingface' in name:
        dataset_name = name.replace('huggingface:', '')
        gen = HuggingfaceGenerator(
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            train_val_test=train_val_test,
            dataset_name=dataset_name)

    else:
        raise NotImplementedError

    if not hasattr(gen, 'in_len'): gen.in_len = maxlen
    if not hasattr(gen, 'out_len'): gen.out_len = maxlen
    if not hasattr(gen, 'vocab_size'): gen.vocab_size = gen.out_dim
    if not hasattr(gen, 'name'): gen.name = name
    if not hasattr(gen, 'timerepeat'): gen.timerepeat = timerepeat
    if not hasattr(gen, 'n_regularizations'): gen.n_regularizations = 0
    gen.output_type = output_type

    return gen


def checkTaskMeanVariance(task_name):
    import pandas as pd
    from tqdm import tqdm
    import numpy as np

    if not os.path.exists(STATSPATH):
        data = {'task_name': [], 'mean': [], 'var': []}

        # create dataframe
        df = pd.DataFrame(data)
        df.to_csv(STATSPATH)
    else:
        df = pd.read_csv(STATSPATH)

    if not task_name in df['task_name'].values:
        gen = Task(batch_size=64, name=task_name, train_val_test='train', steps_per_epoch=None)
        spe = gen.steps_per_epoch

        full_mean = 0
        full_var = 0
        for i in tqdm(range(spe)):
            # idx = 1 if 'wordptb' in task_name else 0
            idx = 0
            batch = gen.__getitem__(i)[0][idx]
            mean_batch = np.mean(np.mean(np.mean(batch, axis=2), axis=1), axis=0)
            var_batch = np.mean(np.mean(np.std(batch, axis=2) ** 2, axis=1), axis=0)

            full_mean += mean_batch
            full_var += var_batch

        full_mean /= spe
        full_var /= spe
        new_row = {'task_name': task_name, 'mean': full_mean, 'var': full_var}

        df = df.append(new_row, ignore_index=True)
        df.to_csv(STATSPATH)
    else:
        full_mean = df.loc[df.task_name == task_name, 'mean'].values[0]
        full_var = df.loc[df.task_name == task_name, 'var'].values[0]

    return full_mean, full_var


if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm

    tasks = ['sl_mnist', 'heidelberg']
    for task_name in tasks:
        print('-' * 50)
        print(task_name)
        full_mean, full_var = checkTaskMeanVariance(task_name)
        print('mean {}, var {}'.format(full_mean, full_var))

    # heidelberg
    # mean 0.04388437186716791, var 0.04195531663882064
    # sl_mnist
    # mean 0.0036727774199098347, var 0.0036592709210085173
    # wordptb
    # mean 0.00010001000191550702, var 0.00010000323345397739
