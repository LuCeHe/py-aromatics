import tensorflow as tf
import keras

class FuncOnInitializer(keras.initializers.Initializer):

    def __init__(self, func, initializer):
        self.func = func
        self.initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        return self.func(self.initializer.__call__(shape, dtype, **kwargs))

    def get_config(self):
        return {'func': self.func, 'initializer': self.initializer}
