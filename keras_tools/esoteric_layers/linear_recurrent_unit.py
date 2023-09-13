# original https://arxiv.org/pdf/2303.06349.pdf


import tensorflow as tf

from pyaromatics.keras_tools.esoteric_layers.geglu import GEGLU


class ComplexGlorotNormal(tf.keras.initializers.Initializer):

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, dtype=None):
        r = tf.keras.initializers.GlorotNormal(self.seed)(shape, dtype=tf.float32) / tf.sqrt(2.)
        i = tf.keras.initializers.GlorotNormal(self.seed)(shape, dtype=tf.float32) / tf.sqrt(2.)
        return tf.dtypes.complex(r, i)

    def get_config(self):  # To support serialization
        return {'seed': self.seed}


class InitFromTensor(tf.keras.initializers.Initializer):

    def __init__(self, tensor):
        self.tensor = tensor

    def __call__(self, shape, dtype=None):
        return self.tensor

    def get_config(self):  # To support serialization
        return {'tensor': self.tensor}


class LinearRecurrentUnitCell(tf.keras.layers.Layer):

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, num_neurons=None, kernel_initializer='orthogonal',
                 rmax=.99, rmin=.4, reduced_phase=True, locked_gamma=False, **kwargs):
        super().__init__(**kwargs)

        self.dtype_ = tf.complex64
        self.init_args = dict(num_neurons=num_neurons, kernel_initializer=kernel_initializer, rmax=rmax, rmin=rmin,
                              locked_gamma=locked_gamma, reduced_phase=reduced_phase)
        self.__dict__.update(self.init_args)

        self.state_size = (num_neurons,)

    def build(self, input_shape):
        n_in = input_shape[-1]
        n_rec = self.num_neurons

        self.C = self.add_weight(shape=(n_rec, n_rec), initializer=ComplexGlorotNormal(), name='C', dtype=self.dtype_)
        self.B = self.add_weight(shape=(n_rec, n_in), initializer=ComplexGlorotNormal(), name='B', dtype=self.dtype_)
        self.D = self.add_weight(shape=(n_in,), initializer=tf.keras.initializers.GlorotNormal(), name='D')

        numax = tf.math.log(-tf.math.log(self.rmin))
        numin = tf.math.log(-tf.math.log(self.rmax))
        nuinit = tf.keras.initializers.RandomUniform(minval=numin, maxval=numax, seed=None)
        self.nu = self.add_weight(shape=(n_rec,), initializer=nuinit, name='nu')

        if self.reduced_phase:
            theta_initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=3.14 / 10, seed=None)
        else:
            theta_initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=2 * 3.14, seed=None)

        self.theta = self.add_weight(shape=(n_rec,), initializer=theta_initializer, name='theta')

        # Normalization
        lambda_ = tf.exp(tf.dtypes.complex(-tf.exp(self.nu), self.theta))
        gamma = tf.sqrt(1 - tf.abs(lambda_) ** 2)
        if self.locked_gamma:
            self.gamma = gamma
        else:
            gamma_initializer = InitFromTensor(gamma)
            self.gamma = self.add_weight(shape=(n_rec,), initializer=gamma_initializer, name='gamma')

        self.built = True

    def call(self, inputs, states, training=None):
        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        u = inputs
        x = states[0]

        lambda_ = tf.exp(tf.dtypes.complex(-tf.exp(self.nu), self.theta))
        Lambda = tf.linalg.diag(lambda_)

        # turning floats to complex
        u_ = tf.cast(u, self.dtype_)
        x_ = tf.cast(x, self.dtype_)
        gamma_ = tf.cast(self.gamma, self.dtype_)

        # rnn operations
        x = x_ @ Lambda + gamma_ * u_ @ self.B
        y = tf.math.real(x_ @ self.C) + self.D * u

        output = y
        new_state = (x,)
        return output, new_state


class ResLRUCell(tf.keras.layers.Layer):

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, num_neurons=None, kernel_initializer='orthogonal',
                 rmax=.99, rmin=.4, reduced_phase=True, dop=.1, locked_gamma=False, **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(num_neurons=num_neurons, kernel_initializer=kernel_initializer, rmax=rmax, rmin=rmin,
                              locked_gamma=locked_gamma, reduced_phase=reduced_phase, dop=dop)
        self.__dict__.update(self.init_args)

        self.state_size = (num_neurons,)

        self.lru = LinearRecurrentUnitCell(
            num_neurons=num_neurons, kernel_initializer=kernel_initializer,
            rmax=rmax, rmin=rmin, reduced_phase=reduced_phase, locked_gamma=locked_gamma
        )
        self.norm = tf.keras.layers.LayerNormalization()
        self.glu = GEGLU(num_neurons, num_neurons, activation='sigmoid', comments='onlyglu')
        self.dropout = tf.keras.layers.Dropout(dop)

    def call(self, inputs, states, training=None):
        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        u = inputs
        x = states[0]

        y, x = self.lru(u, (x,))
        y = self.norm(y)
        y = self.glu(y)
        y = self.dropout(y)

        output = y + u
        new_state = (x,)
        return output, new_state


if __name__ == '__main__':
    num_neurons = 2
    batch_size = 3
    input_tensor = tf.random.normal((batch_size, num_neurons))
    init_state = tf.random.normal((batch_size, num_neurons))
    lru = LinearRecurrentUnitCell(num_neurons=num_neurons)
    lru.build((None, num_neurons))
    out = lru(input_tensor, (init_state,))
    print(out[0].shape)

    reslru = ResLRUCell(num_neurons=num_neurons)
    reslru.build((None, num_neurons))
    out = reslru(input_tensor, (init_state,))
    print(out[0].shape)
