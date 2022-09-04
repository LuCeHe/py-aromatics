import tensorflow as tf

def doubleexp(x, a, b, c, d, e, f, g, h, i, l, m):
    a, b, c, d, e, f, g, h, i, l, m = abs(a), abs(b), abs(c), abs(d), abs(e), abs(f), abs(g), abs(h), abs(i), abs(l), m
    x = x + m
    return \
        (a * tf.exp(b * x) + e * 1 / (1 + f * tf.abs(x) ** (1 + i))) * np.heaviside(-x, .5) \
        + (c * tf.exp(-d * x) + h * 1 / (1 + g * tf.abs(x) ** (1 + l))) * np.heaviside(x, .5)  # one of best so far


def movedgauss(x, a, b, c, d, e, f, g, h, i, l, m):
    a, b, c, d, e, f, g, h, i, l, m = abs(a), abs(b), abs(c), abs(d), abs(e), abs(f), abs(g), abs(h), abs(i), abs(l), m
    x = x + m
    return a * tf.exp(-b * x ** 2)


def movedsigmoid(x, a, b, c, d, e, f, g, h, i, l, m):
    a, b, c, d, e, f, g, h, i, l, m = abs(a), abs(b), abs(c), abs(d), abs(e), abs(f), abs(g), abs(h), abs(i), abs(l), m
    x = x + m
    return a * tf.nn.sigmoid(b * x) * (1 - tf.nn.sigmoid(b * x))


def movedfastsigmoid(x, a, b, c, d, e, f, g, h, i, l, m):
    a, b, c, d, e, f, g, h, i, l, m = abs(a), abs(b), abs(c), abs(d), abs(e), abs(f), abs(g), abs(h), abs(i), abs(l), m
    x = x + m
    return a * 1 / (1 + tf.abs(b * x)) ** 2


def movedfasttail(x, a, b, c, d, e, f, g, h, i, l, m):
    a, b, c, d, e, f, g, h, i, l, m = abs(a), abs(b), abs(c), abs(d), abs(e), abs(f), abs(g), abs(h), abs(i), abs(l), m
    x = x + m
    return a * 1 / (1 + tf.abs(b * x)) ** (1 + tf.abs(c))


def get_asg_shape(shape_name):
    if 'doubleexp' in shape_name:
        asgshape = doubleexp
    elif 'movedgauss' in shape_name:
        asgshape = movedgauss
    elif 'movedsigmoid' in shape_name:
        asgshape = movedsigmoid
    elif 'movedfastsigmoid' in shape_name:
        asgshape = movedfastsigmoid
    elif 'movedfasttail' in shape_name:
        asgshape = movedfasttail
    else:
        asgshape = movedgauss

    return asgshape
