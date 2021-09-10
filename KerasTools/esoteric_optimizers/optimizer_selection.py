import tensorflow as tf
import tensorflow_addons as tfa

from GenericTools.KerasTools.learning_rate_schedules import AddWarmUpToSchedule, DummyConstantSchedule
from GenericTools.KerasTools.esoteric_optimizers.AdaBelief import AdaBelief
from GenericTools.KerasTools.esoteric_optimizers.AdamW import AdamW


def get_optimizer(optimizer_name, lr, lr_schedule='', total_steps=None, weight_decay=False, clipnorm=False,
                  exclude_from_weight_decay=[], warmup_steps=None):
    learning_rate = lr
    if 'cosine_no_restarts' in lr_schedule:
        learning_rate = tf.keras.experimental.CosineDecay(learning_rate, decay_steps=int(4 * total_steps / 5), alpha=.5)
    elif 'cosine_restarts' in lr_schedule:
        learning_rate = tf.keras.experimental.CosineDecayRestarts(learning_rate,
                                                                  first_decay_steps=int(total_steps / 6.5), alpha=.1)
    else:
        # learning_rate = DummyConstantSchedule(learning_rate)
        pass

    if 'warmup' in lr_schedule:
        if warmup_steps is None:
            warmup_steps = total_steps / 6
        learning_rate = AddWarmUpToSchedule(learning_rate, warmup_steps=warmup_steps)

    ma = lambda x: tfa.optimizers.MovingAverage(x) if 'MA' in optimizer_name else x
    optimizer_name = optimizer_name.replace('MA', '')

    swa = lambda x: tfa.optimizers.SWA(x) if 'SWA' in optimizer_name else x
    optimizer_name = optimizer_name.replace('SWA', '')

    if optimizer_name == 'AdamW':
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm,
                          exclude_from_weight_decay=['embedding'] + exclude_from_weight_decay, remove_nans=['all'])
    elif optimizer_name == 'AdaBelief':
        optimizer = AdaBelief(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm,
                              exclude_from_weight_decay=['embedding'] + exclude_from_weight_decay, remove_nans=['all'])
    elif optimizer_name == 'AdaBeliefM1':
        optimizer = AdaBelief(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm,
                              exclude_from_weight_decay=['embedding'], remove_nans=['all'], remove_mean=1)
    elif optimizer_name == 'AdaBeliefM0':
        optimizer = AdaBelief(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm,
                              exclude_from_weight_decay=['embedding'], remove_nans=['all'], remove_mean=0)
    elif optimizer_name == 'AdaBeliefNoise':
        optimizer = AdaBelief(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm,
                              exclude_from_weight_decay=['embedding'], remove_nans=['all'], weight_noise=.075)
    elif optimizer_name == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=clipnorm, momentum=0.99)

    elif optimizer_name == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, clipnorm=clipnorm)
    elif optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    else:
        raise NotImplementedError

    return swa(ma(optimizer))
