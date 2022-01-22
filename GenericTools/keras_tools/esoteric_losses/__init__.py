import sys

from GenericTools.keras_tools.esoteric_losses.advanced_losses import *
from GenericTools.keras_tools.esoteric_losses.forecasting import *
from GenericTools.keras_tools.esoteric_losses.other_losses import *
from tensorflow.keras.losses import mean_squared_error

thismodule = sys.modules[__name__]

def get_loss(initializer_name='sparse_perplexity'):
    loss = getattr(thismodule, initializer_name)
    return loss


# if __name__ == '__main__':
