import tensorflow as tf


class RandomSwitch(tf.keras.layers.Layer):

    def __init__(self, switches_probabilities, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(switches_probabilities, list)
        assert abs(sum(switches_probabilities) - 1) < 1e-7
        self.switches_probabilities = switches_probabilities

    def call(self, inputs, *args, **kwargs):
        assert isinstance(inputs, list)
        b = tf.shape(inputs[0])[0]
        ts = tf.concat([tf.expand_dims(t, -1) for t in inputs], -1)
        output_shape = 'btf' if len(ts.shape) == 4 else 'bt'
        switch = tf.random.categorical(tf.math.log([self.switches_probabilities]), b)
        hot_switch = tf.squeeze(tf.one_hot(switch, len(self.switches_probabilities)), 0)

        switched = tf.einsum('{}o,bo->{}'.format(output_shape, output_shape), ts, hot_switch)
        return switched

    def get_config(self):
        config = {
            'switches_probabilities': self.switches_probabilities,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    b, t, f = 1, 3, 1
    n_ts = 3
    ts = [tf.random.uniform((b, t, f)) for _ in range(n_ts)]
    switches_probabilities = [.1, .3, .6]

    # a = RandomSwitch(switches_probabilities)([t1, t2, t3])
    # expanded_ts = [tf.expand_dims(t, -1) for t in ts]
    t = tf.concat([tf.expand_dims(t, -1) for t in ts], -1)
    print(t.shape)
    print(t)
    for i in range(10):
        a = RandomSwitch(switches_probabilities)(ts)
        print(i)
        print(a)
    # print(t1)
    # print(t2)
    # print(t3)
    # print(a)
