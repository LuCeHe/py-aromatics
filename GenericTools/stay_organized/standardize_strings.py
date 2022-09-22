
def shorten_losses(name):
    shorter_name = name.replace('sparse_', '').replace('categorical_', '').replace('accuracy', 'acc').replace(
        'crossentropy', 'xe').replace('perplexity', 'ppl')
    # shorter_name = name
    return shorter_name