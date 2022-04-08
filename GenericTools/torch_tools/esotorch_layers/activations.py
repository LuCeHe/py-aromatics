import torch
from torch import nn


def CritialVar(name, force_value=None):
    c = 1
    if name == 'relu':
        c = 2
    elif name == 'requ':
        c = 1 / 3
    elif name in ['re07u', 'grepu', 'bigrepu']:
        c = 1.
    else:
        raise NotImplementedError

    if not force_value is None:
        c = force_value

    return c


def ActivationRedirection(name, power=1.1, power2=1.391948):
    activation = None
    n = 1
    if ':' in name:
        old_name = name
        name = old_name.split(':')[0]
        n = int(old_name.split(':')[1])

    if name == 'relu':
        activation = nn.ReLU()
    elif name == 'repu':
        activation = RePU(1, power)
    elif name == 'grepu':
        if n == 0:
            activation = RePU(1, power)
        else:
            activation = gRePU(1, power2)

    elif name == 'bigrepu':
        if n == 0:
            activation = gRePU(1, power)
        else:
            activation = gRePU(1, power2)
    else:
        raise NotImplementedError

    return activation


# class Activation(torch.nn.Module):
#     def __init__(self, name):
#         super().__init__()
#
#     def forward(self, x):
#         return torch.pow(self.slope * x, self.exponent)


class RePU(torch.nn.Module):
    def __init__(self, slope=1., power=2.):
        super(RePU, self).__init__()
        self.slope = slope
        self.exponent = power

    def forward(self, x):
        return torch.pow(self.slope * x.relu(), self.exponent)


class gRePU(torch.nn.Module):
    def __init__(self, slope=1., power=2., var=0.0001):
        super().__init__()
        self.slope = slope
        self.exponent = power
        self.var = var

    def forward(self, x):
        return (1 - torch.exp(-x.relu() ** 2 / (2 * self.var))) * torch.pow(self.slope * x.relu(), self.exponent)


class CSwish(torch.nn.Module):
    def __init__(self, slope=1, power=1):
        super(CSwish, self).__init__()
        self.slope = torch.tensor(slope)
        self.exponent = torch.tensor(power)

    def forward(self, x):
        return torch.pow(x * torch.sigmoid(x / self.slope), self.exponent)


class LRePU(torch.nn.Module):
    def __init__(self, slope=1, power=2):
        super(LRePU, self).__init__()
        self.slope = torch.tensor(slope)
        self.exponent = power * torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.pow(self.slope * x, self.exponent)


class LCSwish(torch.nn.Module):
    def __init__(self, slope=1, power=1):
        super(LControllSwish, self).__init__()
        self.slope = slope * torch.nn.Parameter(torch.ones(1))
        self.exponent = power * torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.pow(x * torch.sigmoid(x / self.slope), self.exponent)
