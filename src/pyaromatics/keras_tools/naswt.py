import tensorflow as tf
import numpy as np

EMPIRICAL_MAX = -2.5e2
EMPIRICAL_MIN = -6.5e6


def corrdistintegral_eval_score(upp):
    def fun(jacob, labels=None):
        xx = jacob.reshape(jacob.size(0), -1).detach().cpu().numpy()
        corrs = np.corrcoef(xx)
        return np.logical_and(corrs < upp, corrs > 0).sum()
    return fun

def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v = np.linalg.eigvals(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1. / (v + k))


def IWPJS(model, batch, return_dict=None):
    # inv_weighted_parameters_jacobian_score
    x, y = batch
    # define the loss inside here
    with tf.GradientTape(persistent=True) as tape:
        outputs = model(x)
        print([l.shape for l in y])
        print([type(l) for l in y])
        print([l.shape for l in outputs])
        #l = model.loss([y, outputs])
        l = model.compiled_loss(y, outputs)
        # l = model.losses(y, outputs)
    variables = model.trainable_variables
    parameters_grads = tape.gradient(l, variables)  # (outputs, variables)
    len_p = len(parameters_grads)

    s = 0
    j = 0
    for i, g in enumerate(reversed(parameters_grads)):
        if not g is None:
            if len(g.shape) > 1:
                batch_size = g.shape[0]
                if batch_size > 1:
                    j += 1
                    try:
                        g = g.numpy().reshape(batch_size, -1)
                        s += j * eval_score(g, None)
                    except:
                        pass

    if s == 0:
        s = np.nan
    else:
        normalizer_weights = len_p * (len_p + 1) / 2
        s /= j * normalizer_weights

        s = 1 - (s-EMPIRICAL_MIN) / (EMPIRICAL_MAX-EMPIRICAL_MIN)

    if not return_dict is None:
        return_dict['score'] = s.real
    return s.real



def JS(model, batch, return_dict=None):
    # inv_weighted_parameters_jacobian_score

    x = batch[0]
    if isinstance(x, tuple):
        batch_size = x[0].shape[0]
    else:
        batch_size = x.shape[0]

    inp = [tf.Variable(b, dtype=tf.float32, name=mi.name[:-4]) for b, mi in zip(batch[0], model.inputs)]
    print([i.name for i in inp])
    print(len(inp))
    print(model.inputs)
    print()
    exit
    with tf.GradientTape(persistent=True) as tape:
        outputs = model(batch)
        # print(outputs)
        l = tf.keras.losses.CategoricalCrossentropy()(outputs, batch[0][-1])
        print(l)
        #l = model.loss(y, outputs)


        # l = model.evaluate(batch)
        # l = model.compiled_loss(y, outputs)
        # l = model.losses(y, outputs)

    jacob = tape.gradient(l, model.inputs)  # (outputs, variables)

    print(jacob)
    jacob = jacob.numpy().reshape(batch_size, -1)
    s = corrdistintegral_eval_score(0.25)(jacob)
    return s.real
