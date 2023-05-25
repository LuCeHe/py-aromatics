
class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 20
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20

    def __init__(self, T=-1):
        if not T < 0:
            self.num_steps = T

class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20

    def __init__(self, T=-1):
        if not T < 0:
            self.num_steps = T

class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20

    def __init__(self, T=-1):
        if not T < 0:
            self.num_steps = T

class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 100
    max_epoch = 5
    max_max_epoch = 5
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 3
    epoch_size = 10

    def __init__(self, T=-1):
        if not T < 0:
            self.num_steps = T


def get_config(model, T=-1):
    """Get model config."""
    config = None
    if model == "small":
        config = SmallConfig(T)
    elif model == "medium":
        config = MediumConfig(T)
    elif model == "large":
        config = LargeConfig(T)
    elif model == "test":
        config = TestConfig(T)
    else:
        raise ValueError("Invalid model: %s", model)
    return config