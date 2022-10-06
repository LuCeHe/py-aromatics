
def shorten_losses(name):
    shorter_name = name.replace('sparse_', '').replace('categorical_', '').replace('accuracy', 'acc').replace(
        'crossentropy', 'xe').replace('perplexity', 'ppl')
    shorter_name = shorter_name.replace('deterministic', 'det').replace('False', 'F').replace('True', 'T').replace(',', '')
    shorter_name = shorter_name.replace('_', ' ').replace(' ', '')
    return shorter_name