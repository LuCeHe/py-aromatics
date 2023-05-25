from GenericTools.torch_tools.esoteric_tasks.csv import CsvTask
from GenericTools.torch_tools.esoteric_tasks.ptb import PennTreeBankTask


def TorchTask(task_name, batch_size, epochs, steps_per_epoch, maxlen, data_split, datadir=None, string_config='', **kwargs):
    assert data_split in ['train', 'valid', 'validation', 'val', 'test']

    if task_name == 'ptb':
        gen = PennTreeBankTask(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, maxlen=maxlen,
                               data_split=data_split, datadir=datadir, string_config=string_config)
    elif task_name == 'csv':
        gen = CsvTask(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, maxlen=maxlen, **kwargs)

    else:
        raise NotImplementedError

    return gen
