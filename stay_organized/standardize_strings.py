
def shorten_losses(name):
    shorter_name = name.replace('sparse_', '').replace('categorical_', '').replace('accuracy', 'acc').replace(
        'crossentropy', 'xe').replace('perplexity', 'ppl')
    shorter_name = shorter_name.replace('deterministic', 'det').replace('False', 'F').replace('True', 'T').replace(',', '')
    shorter_name = shorter_name.replace('learning_rate', 'lr')
    shorter_name = shorter_name.replace('steps_per_epoch', 'spe')
    shorter_name = shorter_name.replace('epoch', 'ep')
    shorter_name = shorter_name.replace('layers', 'depth')
    shorter_name = shorter_name.replace('pretrain', 'pre')
    shorter_name = shorter_name.replace('activation', 'act')
    shorter_name = shorter_name.replace('initial', 'i')
    shorter_name = shorter_name.replace('final', 'f')
    shorter_name = shorter_name.replace('max', 'M')
    shorter_name = shorter_name.replace('min', 'm')
    shorter_name = shorter_name.replace('attention_head_count', 'mh')
    shorter_name = shorter_name.replace('_preprocessinput', '')
    shorter_name = shorter_name.replace('deslice', 'dsl')
    shorter_name = shorter_name.replace('task_name', 'task')
    shorter_name = shorter_name.replace('net_name', 'net')
    shorter_name = shorter_name.replace('_mode_', '_^')
    shorter_name = shorter_name.replace('LSC_norms', 'LSC')
    shorter_name = shorter_name.replace('mean', 'a')

    return shorter_name