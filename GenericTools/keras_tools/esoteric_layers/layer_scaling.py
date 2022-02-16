import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import array_ops

from tensorflow.python.keras import constraints


class LayerScaling(Layer):

    def __init__(self,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 grad_through_maxmin=True,
                 **kwargs):
        super(LayerScaling, self).__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError('Expected an int or a list/tuple of ints for the '
                            'argument \'axis\', but received: %r' % axis)

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        self.supports_masking = True
        self.grad_through_maxmin = grad_through_maxmin

        self.grad_switch = tf.stop_gradient
        if not grad_through_maxmin:
            self.grad_switch = lambda x: x

        maxs = lambda x: self.grad_switch(tf.reduce_max(x, axis=self.axis, keepdims=True))
        mins = lambda x:self.grad_switch(tf.reduce_min(x, axis=self.axis, keepdims=True))
        if center =='maxmin':
            self.centralize = lambda x: (maxs(x) + mins(x)) / 2
        elif center == 'mean':
            self.centralize = lambda x: tf.reduce_mean(x, axis=self.axis, keepdims=True)
        else:
            self.centralize = lambda x: (maxs(x) + mins(x)) / 2

        if scale =='maxmin':
            self.scalerize = lambda x: (maxs(x) - mins(x))
        elif scale == 'std':
            self.scalerize = lambda x: tf.math.reduce_std(x, axis=self.axis, keepdims=True)
        else:
            self.scalerize = lambda x: (maxs(x) - mins(x))


    def build(self, input_shape):

        param_shape = input_shape[self.axis]
        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.beta = None

        self.built = True

    def call(self, inputs):
        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)

        # Broadcasting only necessary for norm when the axis is not just
        # the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis] = input_shape.dims[self.axis].value

        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and self.axis != [ndims - 1]):
                return array_ops.reshape(v, broadcast_shape)
            return v

        center = self.centralize(inputs)
        width = (self.scalerize(inputs) + self.epsilon)
        outputs = inputs - center
        outputs /= width

        gamma = _broadcast(self.gamma)
        beta = _broadcast(self.beta)

        outputs = gamma * outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'grad_through_maxmin': self.grad_through_maxmin
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    t = tf.random.normal((2, 3, 4))
    print(t)

    o = LayerScaling(axis=0)(t)
    print(o)
    o = LayerScaling(axis=1)(t)
    print(o)
    o = LayerScaling(axis=2)(t)
    print(o)