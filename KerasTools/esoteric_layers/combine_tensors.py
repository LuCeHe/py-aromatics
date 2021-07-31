import tensorflow as tf


class CombineTensors(tf.keras.layers.Layer):
    def __init__(self, n_tensors, axis=None, sigmoidal_gating=False, initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.n_tensors = n_tensors
        self.axis = axis
        self.initializer = initializer
        self.sigmoidal_gating = sigmoidal_gating
        self.s = tf.math.sigmoid if sigmoidal_gating else lambda x: x

    def build(self, input_shape):
        axis_size = input_shape[0][self.axis] if not self.axis is None else 1
        self.matrices = []
        for i in range(self.n_tensors):
            m = self.add_weight(shape=(axis_size,), initializer=self.initializer, name='fd_{}'.format(i))
            self.matrices.append(m)

        self.built = True

    def call(self, inputs, training=None):
        output = 0
        for i, m in zip(inputs, self.matrices):
            output += i * self.s(m)

        return output

    def get_config(self):
        config = {
            'n_tensors': self.n_tensors,
            'sigmoidal_gating': self.sigmoidal_gating,
            'initializer': self.initializer,
            'axis': self.axis,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    n_tensors = 3
    input_tensors = [tf.random.uniform((2, 3, 10))] * n_tensors
    ct = CombineTensors(n_tensors=n_tensors, sigmoidal_gating=False, axis=1)
    f = ct(input_tensors)
    print(f.shape)

    inpl = [tf.keras.layers.Input(t.shape[1:]) for t in input_tensors]
    fl = ct(inpl)
    model = tf.keras.models.Model(inpl, fl)
    model.compile('SGD', 'categorical_crossentropy')
    model.fit(input_tensors, input_tensors[0])
    model.summary()
