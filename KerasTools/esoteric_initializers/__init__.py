import sys, inspect
from GenericTools.KerasTools.esoteric_initializers.extra_variance_criticality import *

# from tensorflow.keras.initializers import *

esoteric_initializers_list = [
    'BiGamma', 'BiGamma10', 'BiGammaOrthogonal', 'CauchyOrthogonal', 'GlorotBiGamma', 'HeBiGamma',
    'GlorotBiGammaOrthogonal',
    'GlorotCauchyOrthogonal', 'GlorotOrthogonal', 'GlorotTanh', 'MoreVarianceScalingAndOrthogonal', 'NoZeroGlorot',
    'NoZeroGlorotOrthogonal', 'TanhBiGamma10', 'TanhBiGamma10Orthogonal'
]


def print_classes():
    cls = ''
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            cls += name + "','"
    print(cls)


# print_classes()

thismodule = sys.modules[__name__]


def get_initializer(initializer_name='GlorotUniform'):
    initializer = getattr(thismodule, initializer_name)()
    return initializer
