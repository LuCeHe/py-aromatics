import os

from GenericTools.KerasTools.esoteric_tasks.random_task import RandomTask
from GenericTools.KerasTools.esoteric_tasks.wizard_of_wikipedia import WikipediaWizardGenerator
from GenericTools.StayOrganizedTools.utils import str2val
#
# FILENAME = os.path.realpath(__file__)
# CDIR = os.path.dirname(FILENAME)
# STATSPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'task_stats.csv'))


def Task(batch_size=64, steps_per_epoch=None, epochs=1, task_name='wow', data_split='train',
         maxlen=100, string_config='', data_path=None, shuffle=True):

    # assert data_split in ['train', 'test', 'val', 'validation']

    if task_name == 'wow':
        encoder_maxlen = str2val(string_config, 'encodermaxlen', int, default=maxlen)
        decoder_maxlen = str2val(string_config, 'decodermaxlen', int, default=maxlen)
        n_dialogues = str2val(string_config, 'ndialogues', int, default='full')

        gen = WikipediaWizardGenerator(
            data_path=data_path, n_dialogues=n_dialogues, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
            encoder_maxlen=encoder_maxlen, decoder_maxlen=decoder_maxlen, epochs=epochs,
            tokenizer_choice='bpe', data_split=data_split, shuffle=shuffle)

    elif task_name == 'random':
        # example_task_name = 'random_intypes:[int,int,float]_inshapes:[(3,),(3,),(3,)]' \
        #                     '_outtypes:[int]_outshapes:[(3,)]'
        gen = RandomTask(
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            string_config=string_config,)

    else:
        raise NotImplementedError

    if not hasattr(gen, 'in_len'): gen.in_len = maxlen
    if not hasattr(gen, 'out_len'): gen.out_len = maxlen
    if not hasattr(gen, 'task_name'): gen.name = task_name
    if not hasattr(gen, 'epochs'): gen.epochs = epochs
    if not hasattr(gen, 'steps_per_epoch'): gen.steps_per_epoch = steps_per_epoch

    return gen
