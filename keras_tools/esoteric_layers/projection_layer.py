import tensorflow as tf

class ProjectionLayer(tf.keras.layers.Layer):
    def __init__(self):
        # model hyper parameter variables
        super().__init__()
        self.project_matrix = 'project_matrix'

    def call(self, inputs, **kwargs):
        output = inputs @ self.project_matrix

        return output