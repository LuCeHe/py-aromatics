from GenericTools.KerasTools.esoteric_tasks.bit_generator import OneBitTimeDependentGenerator
from GenericTools.KerasTools.esoteric_tasks.grammar_generators import CFG_AutoEncoding_Generator, MergeSearch
from GenericTools.KerasTools.esoteric_tasks.heidelberg_generator import SpokenHeidelbergDigits
from GenericTools.KerasTools.esoteric_tasks.huggingface_generator import HuggingfaceGenerator
from GenericTools.KerasTools.esoteric_tasks.mnist_generators import SeqMNIST
from GenericTools.KerasTools.esoteric_tasks.monkey_generator import MonkeyGenerator
from GenericTools.KerasTools.esoteric_tasks.ptb_generator import PTBGenerator
from GenericTools.KerasTools.esoteric_tasks.random_generator import RandomGenerator
from GenericTools.KerasTools.esoteric_tasks.xor import XorGenerator


def Task(n_dt_per_step=1, batch_size=64, steps_per_epoch=1, epochs=1, name='time_ae', train_val_test='train',
         neutral_phase_length=0, category_coding='onehot', inherit_from_gen=False, maxlen=100, old=False):
    if name == 'time_ae':
        gen = CFG_AutoEncoding_Generator(
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            time_period=n_dt_per_step,
            maxlen=maxlen)

    elif name == 'onebit':
        gen = OneBitTimeDependentGenerator(
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            timesteps_dependency=3,
            maxlen=maxlen,
            neutral_phases=True,
            repetitions=n_dt_per_step)

    elif 'ptb' == name:
        gen = PTBGenerator(
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen,
            repetitions=n_dt_per_step,
            train_val_test=train_val_test,
            neutral_phase_length=neutral_phase_length,
            category_coding=category_coding)

    elif 'wordptb' == name:
        gen = PTBGenerator(
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen,
            repetitions=n_dt_per_step,
            train_val_test=train_val_test,
            neutral_phase_length=neutral_phase_length,
            category_coding=category_coding,
            char_or_word='word')

    elif 'wiki103' == name:
        gen = Wiki103Generator(
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            maxlen=maxlen,
            repetitions=n_dt_per_step,
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
            time_period=n_dt_per_step,
            maxlen=maxlen)

    elif name == 's_mnist':
        gen = SeqMNIST(
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            n_dt_per_step=n_dt_per_step,
            permuted=False)

    elif name == 'ps_mnist':
        gen = SeqMNIST(
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            n_dt_per_step=n_dt_per_step,
            permuted=True,
            inherit_from_gen=inherit_from_gen)

    elif name == 'heidelberg':
        gen = SpokenHeidelbergDigits(
            epochs=epochs,
            batch_size=batch_size,
            tvt=train_val_test,
            steps_per_epoch=steps_per_epoch,
            n_dt_per_step=n_dt_per_step)

    elif name == 'wmt14':
        gen = Wmt14Generator(
            epochs=epochs,
            batch_size=batch_size,
            train_val_test=train_val_test,
            steps_per_epoch=steps_per_epoch,
            repetitions=n_dt_per_step,
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
            steps_per_epoch=steps_per_epoch)

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
    if not hasattr(gen, 'name'): gen.name = name
    if not hasattr(gen, 'n_dt_per_step'): gen.n_dt_per_step = n_dt_per_step
    if not hasattr(gen, 'n_regularizations'): gen.n_regularizations = 0
    gen.old = old

    return gen
