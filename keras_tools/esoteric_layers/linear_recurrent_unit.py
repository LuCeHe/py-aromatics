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
                 rmax=.95, rmin=.4, reduced_phase=True, locked_gamma=False, **kwargs):
        super().__init__(**kwargs)

        self.dtype_ = tf.complex64
        self.init_args = dict(num_neurons=num_neurons, kernel_initializer=kernel_initializer, rmax=rmax, rmin=rmin,
                              locked_gamma=locked_gamma, reduced_phase=reduced_phase)
        self.__dict__.update(self.init_args)

        self.state_size = (num_neurons,)

    def build(self, input_shape):

        print('input_shape', input_shape)
        n_in = input_shape[-1]
        n_rec = self.num_neurons

        self.C = self.add_weight(shape=(n_rec, n_rec), initializer=ComplexGlorotNormal(), name='C', dtype=self.dtype_)
        self.B = self.add_weight(shape=(n_in, n_rec), initializer=ComplexGlorotNormal(), name='B', dtype=self.dtype_)
        self.D = self.add_weight(shape=(n_in,), initializer=tf.keras.initializers.RandomNormal(stddev=1), name='D')

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

    def call(self, inputs, states, **kwargs):
        if not kwargs['training'] is None:
            tf.keras.backend.set_learning_phase(kwargs['training'])

        u = inputs
        x = states[0]

        lambda_ = tf.exp(tf.dtypes.complex(-tf.exp(self.nu), self.theta))
        Lambda = tf.linalg.diag(lambda_)

        # turning floats to complex
        u_ = tf.cast(u, self.dtype_)
        x_ = tf.cast(x, self.dtype_)
        gamma_ = tf.cast(self.gamma, self.dtype_)

        # rnn operations
        print('u_', u_.shape)
        print('B', self.B.shape)
        new_u = gamma_ * u_ @ self.B
        new_x_ = tf.einsum('bi,ij->bj', x_, Lambda)

        x_ = new_x_ + new_u

        y = tf.math.real(x_ @ self.C) + self.D * u
        output = y
        new_state = (x_,)
        return output, new_state


class ResLRUCell(tf.keras.layers.Layer):

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, num_neurons=None, kernel_initializer='orthogonal',
                 rmax=.95, rmin=.4, reduced_phase=True, dop=.1, locked_gamma=False, **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(num_neurons=num_neurons, kernel_initializer=kernel_initializer, rmax=rmax, rmin=rmin,
                              locked_gamma=locked_gamma, reduced_phase=reduced_phase, dop=dop)
        self.__dict__.update(self.init_args)

        self.state_size = (num_neurons,)

        self.lru = LinearRecurrentUnitCell(
            num_neurons=num_neurons, kernel_initializer=kernel_initializer,
            rmax=rmax, rmin=rmin, reduced_phase=reduced_phase, locked_gamma=locked_gamma
        )
        # self.lrurnn = tf.keras.layers.RNN(self.lru, return_sequences=True, return_state=True)
        self.norm = tf.keras.layers.LayerNormalization()
        self.glu = GEGLU(num_neurons, num_neurons, activation='sigmoid', comments='onlyglu')
        self.gelu = tf.keras.layers.Activation('gelu')
        self.dropout_1 = tf.keras.layers.Dropout(dop)
        self.dropout_2 = tf.keras.layers.Dropout(dop)
        # self.adapter = lamdba x: x

    def build(self, input_shape):

        new_input_shape = input_shape[:-1] + (self.num_neurons,)
        self.lru.build(new_input_shape)
        self.norm.build(new_input_shape)
        self.glu.build(new_input_shape)

        if input_shape[-1] != self.num_neurons:
            self.adapter = tf.keras.layers.Dense(self.num_neurons, activation='linear')
            self.adapter.build(input_shape)
        else:
            self.adapter = lambda x: x

        self.built = True

    def call(self, inputs, states, **kwargs):
        if not kwargs['training'] is None:
            tf.keras.backend.set_learning_phase(kwargs['training'])
        x = states[0]

        adapted = self.adapter(inputs)
        u = self.norm(adapted)
        y, x = self.lru.call(u, (x,), **kwargs)
        y = self.gelu(y)
        y = self.dropout_1(y)
        y = self.glu(y)
        y = self.dropout_2(y)

        output = y + adapted
        new_state = x
        return output, new_state


# FFN version of LinearRecurrentUnitCell
class LinearRecurrentUnitFFN(tf.keras.layers.Layer):

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, num_neurons=None, kernel_initializer='orthogonal',
                 rmax=.95, rmin=.4, reduced_phase=True, locked_gamma=False, **kwargs):
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
        self.D = self.add_weight(shape=(n_in,), initializer=tf.keras.initializers.RandomNormal(stddev=1), name='D')

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

    def call(self, inputs, training=None):

        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        u = inputs

        lambda_ = tf.dtypes.complex(-tf.exp(self.nu), self.theta)
        lambda_ = tf.repeat(tf.expand_dims(lambda_, axis=0), tf.shape(u)[1], axis=0)

        # repeat on axis 0
        time = tf.range(tf.shape(u)[1], dtype=tf.float32)
        time = tf.cast(tf.expand_dims(time, axis=-1), self.dtype_)

        # exponentiate by time
        lambda_pow = lambda_ * time
        lambda_pow = tf.exp(lambda_pow)
        Lambda_pow = tf.linalg.diag(lambda_pow)

        lambda_minpow = -lambda_ * time
        lambda_minpow = tf.exp(lambda_minpow)
        Lambda_minpow = tf.linalg.diag(lambda_minpow)

        # turning floats to complex
        u_ = tf.cast(u, self.dtype_)
        gamma_ = tf.cast(self.gamma, self.dtype_)

        # rnn operations
        new_u = gamma_ * u_ @ self.B

        x_ = tf.einsum('bti,tij->btj', new_u, Lambda_minpow)
        x_ = tf.cumsum(x_, axis=1)
        print('inside', x_)
        x_ = tf.einsum('bti,tij->btj', x_, Lambda_pow)

        y = tf.math.real(x_ @ self.C) + self.D * u
        output = y
        print('\n\noutside')
        return output


def test_1():
    import time
    # set all seeds

    num_neurons = 210
    time_steps = 100
    batch_size = 100

    test_forward_pass = False
    test_rnn_is_ffn = False
    test_long_time = False
    test_reslru = True

    if test_forward_pass:
        input_tensor = tf.random.normal((batch_size, num_neurons))
        init_state = tf.random.normal((batch_size, num_neurons))
        lru = LinearRecurrentUnitCell(num_neurons=num_neurons)
        out = lru(input_tensor, (init_state,))
        print(out[0].shape)

        reslru = ResLRUCell(num_neurons=num_neurons)
        out = reslru(input_tensor, (init_state,))
        print(out[0].shape)

        input_tensor = tf.random.normal((batch_size, time_steps, num_neurons))
        lruffn = LinearRecurrentUnitFFN(num_neurons=num_neurons)
        out = lruffn(input_tensor)
        print(out.shape)

        lru = LinearRecurrentUnitCell(num_neurons=num_neurons)
        lrurnn = tf.keras.layers.RNN(lru, return_sequences=True)
        out = lrurnn(input_tensor)
        print(out.shape)

    if test_rnn_is_ffn:
        # move parameters from lruffn to lrurnn
        print('=-.-=' * 100)
        print('hey!')
        input_tensor = tf.random.normal((batch_size, time_steps, num_neurons)) / 1

        lruffn = LinearRecurrentUnitFFN(num_neurons=num_neurons)
        lrucell = LinearRecurrentUnitCell(num_neurons=num_neurons)
        lrurnn = tf.keras.layers.RNN(lrucell, return_sequences=True)

        _ = lruffn(input_tensor)
        _ = lrurnn(input_tensor)
        print('-' * 100)

        lrurnn.set_weights(lruffn.get_weights())
        start_time = time.time()
        outrnn = lrurnn(input_tensor)
        rnn_time = time.time() - start_time
        print(outrnn)

        start_time = time.time()
        outffn = lruffn(input_tensor)
        ffn_time = time.time() - start_time
        print(outffn)

        print('rnn time: ', rnn_time)
        print('ffn time: ', ffn_time)

        print(tf.reduce_sum(tf.square(outffn - outrnn) / tf.square(outffn)))

    if test_long_time:
        # move parameters from lruffn to lrurnn
        print('=-.-=' * 100)
        print('hey!')
        input_tensor = tf.random.normal((batch_size, time_steps, num_neurons)) / .1

        lruffn = LinearRecurrentUnitFFN(num_neurons=num_neurons)

        outffn = lruffn(input_tensor)
        print(outffn)

    if test_reslru:
        input_tensor = tf.random.normal((batch_size, time_steps, num_neurons))
        reslru = ResLRUCell(num_neurons=num_neurons)
        # lru = LinearRecurrentUnitCell(num_neurons=num_neurons)
        lrurnn = tf.keras.layers.RNN(reslru, return_sequences=True)
        out = lrurnn(input_tensor)
        print(out.shape)

    # ResLRUCell


if __name__ == '__main__':
    test_1()
