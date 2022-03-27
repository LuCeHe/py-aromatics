from GenericTools.torch_tools.esoteric_tasks.language_modeling import PennTreeBankTask


def TorchTask(task_name, batch_size, epochs, steps_per_epoch, maxlen, train_val_test, datadir=None, string_config=''):
    if task_name == 'ptb':
        gen = PennTreeBankTask(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, maxlen=maxlen,
                               train_val_test=train_val_test, datadir=datadir, string_config=string_config)
    else:
        raise NotImplementedError

    return gen
