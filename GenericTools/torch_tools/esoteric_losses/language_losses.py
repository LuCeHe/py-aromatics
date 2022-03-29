import torch, math

def turnxent(x, y):
    x = torch.swapaxes(x, 1, 2)
    # loss = torch.nn.functional.cross_entropy(x, y, label_smoothing=.1)
    loss = torch.nn.functional.cross_entropy(x, y)
    return loss

def bpc(x, y):
    return turnxent(x, y)/ math.log(2.)


def perplexity(x, y):
    return torch.exp(turnxent(x, y))
