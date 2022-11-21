import tensorflow as tf

class OutInitializer(tf.keras.initializers.Initializer):

    def get_config(self):
        return {
            'init_tensor': self.init_tensor,
        }

    def __init__(self, init_tensor):
        self.init_tensor = init_tensor

    def __call__(self, shape, dtype=tf.float32, **kwargs):

        return self.init_tensor
