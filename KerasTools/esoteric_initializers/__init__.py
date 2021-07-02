import sys, inspect
from GenericTools.KerasTools.esoteric_initializers.glorot_normal_orthogonal_cauchy import *

esoteric_initializers_list = [
    'BiGamma', 'BiGamma10', 'BiGammaOrthogonal', 'CauchyOrthogonal', 'GlorotCauchyOrthogonal', 'GlorotOrthogonal',
    'GlorotTanh', 'MoreVarianceScalingAndOrthogonal', 'TanhBiGamma10', 'TanhBiGamma10Orthogonal',
]


def print_classes():
    cls = ''
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            cls += name + ','
            print(name)
    print(cls)


print_classes()
