import sys

from GenericTools.KerasTools.esoteric_losses.advanced_losses import *

thismodule = sys.modules[__name__]

def get_loss(initializer_name='sparse_perplexity'):
    loss = getattr(thismodule, initializer_name)
    return loss


# if __name__ == '__main__':
