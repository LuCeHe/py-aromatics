import tensorflow as tf
from tensorflow.keras.layers import Dense


class FactorizedDense(tf.keras.layers.Layer):

    def __init__(self, factorized_dims=[1], mask_diagonal=False, initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.factorized_dims, self.mask_diagonal = factorized_dims, mask_diagonal
        self.initializer = initializer

        n = factorized_dims[-1]
        self.mask = tf.ones((n, n)) - tf.eye(n) if mask_diagonal else tf.ones((n, n))

    def build(self, input_shape):
        n_input = input_shape[-1]
        input_dims = [n_input] + self.factorized_dims[:-1]
        output_dims = self.factorized_dims

        self.matrices = []
        for i, (id, od) in enumerate(zip(input_dims, output_dims)):
            f = self.add_weight(shape=(id, od), initializer=self.initializer, name='fd_{}'.format(i))
            self.matrices.append(f)

        self.built = True

    def call(self, inputs, training=None):
        x = inputs

        bm = None
        for m in self.matrices:
            bm = m if bm is None else bm @ m

        bm = bm * self.mask
        x = x @ bm
        return x

    def get_config(self):
        config = {
            'dropword_prob': self.dropword_prob,
            'vocab_size': self.vocab_size,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    input_tensor = tf.random.uniform((2, 3, 10))
    fense = FactorizedDense(factorized_dims=[2, 3, 3, 10], mask_diagonal=True)
    f = fense(input_tensor)
    print(f.shape)

    inpl = tf.keras.layers.Input(input_tensor.shape[1:])
    fl = fense(inpl)
    model = tf.keras.models.Model(inpl, fl)
    model.compile('SGD', 'categorical_crossentropy')
    model.fit(input_tensor, input_tensor)
