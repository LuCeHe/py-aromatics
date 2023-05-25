import tensorflow as tf


class GeneralActivityRegularization(tf.keras.layers.Layer):
    """Layer that applies an update to the cost function based input activity.

    Args:
      l1: L1 regularization factor (positive float).
      l2: L2 regularization factor (positive float).

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    """

    def __init__(self, regularizer_function, **kwargs):
        super().__init__(
            activity_regularizer=regularizer_function, **kwargs
        )
        self.supports_masking = True
        self.regularizer_function = regularizer_function

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"regularizer_function": self.regularizer_function}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))