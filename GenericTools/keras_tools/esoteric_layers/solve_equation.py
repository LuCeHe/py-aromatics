import tensorflow as tf


class SolveEquation(tf.keras.layers.Layer):

    def __init__(self, params, equation, shape=(10,), noisy=False, **kwargs):
        super().__init__(**kwargs)
        self.equation = equation
        self.shape = shape
        self.params = params
        self.constraint = [
            lambda x: tf.abs(x) if p == 'positive'
            else lambda x: x
            for p in params]

        self.noise = lambda x: x + tf.stop_gradient(x * tf.random.normal(x.shape) / 20) if noisy else x

    def build(self, input_shape):
        self.ws = []
        for i in self.params:
            if isinstance(i, str):
                w = self.add_weight(
                    name='w_{}'.format(i), shape=self.shape,
                    initializer=tf.keras.initializers.RandomUniform(minval=.2, maxval=3.),
                    trainable=True
                )
            else:
                w = i
            self.ws.append(w)
        # print(self.ws)
        self.built = True

    def call(self, inputs, **kwargs):
        loss = self.equation(*[self.noise(c(w)) for c, w in zip(self.constraint, self.ws)])
        loss = tf.reduce_mean(tf.boolean_mask(loss, tf.math.is_finite(loss)))

        self.add_loss(tf.reduce_mean(loss))
        self.add_metric(loss, name='cost', aggregation='mean')

        return inputs

    def get_config(self):
        config = {
            'params': self.params, 'equation': self.equation, 'shape': self.shape
        }
        return dict(list(super().get_config().items()) + list(config.items()))
