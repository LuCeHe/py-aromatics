import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils

from tensorflow.keras.layers import PReLU
from tensorflow.python.ops.nn_impl import swish


class RePU(Layer):

    def __init__(self,
                 p_initializer='ones',
                 slope_initializer='ones',
                 p_regularizer=None,
                 p_constraint=None,
                 shared_axes=[1],
                 base_activation='relu',
                 slope=True,
                 trainable_p=True,
                 trainable_slope=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.p_initializer = initializers.get(p_initializer)
        self.p_regularizer = regularizers.get(p_regularizer)
        self.p_constraint = constraints.get(p_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

        self.base_activation = base_activation
        self.slope = slope
        self.slope_initializer = initializers.get(slope_initializer)
        if base_activation == 'relu':
            self.activation = backend.relu
        elif base_activation == 'swish':
            self.activation = swish

        self.trainable_p = trainable_p
        self.trainable_slope = trainable_slope

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1

        self.p = self.add_weight(
            shape=param_shape,
            name='power',
            initializer=self.p_initializer,
            regularizer=self.p_regularizer,
            constraint=self.p_constraint,
            trainable=self.trainable_p
        )
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)

        if self.slope:
            self.s = self.add_weight(
                shape=param_shape,
                name='slope',
                initializer=self.slope_initializer,
                trainable=self.trainable_slope
            )
        else:
            self.s = 1

        self.built = True

    def call(self, inputs):
        pos = self.activation(self.s * inputs) ** self.p
        return pos

    def get_config(self):
        config = {
            'p_initializer': initializers.serialize(self.p_initializer),
            'p_regularizer': regularizers.serialize(self.p_regularizer),
            'p_constraint': constraints.serialize(self.p_constraint),
            'shared_axes': self.shared_axes,
            'slope_initializer': self.slope_initializer,
            'base_activation': self.base_activation,
            'slope': self.slope,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
