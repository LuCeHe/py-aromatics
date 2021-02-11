import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.distribute import sharded_variable
import tensorflow.keras.backend as K

from GenericTools.KerasTools.esoteric_layers.convenience_layers import OneHot

"""
sources:
    - https://github.com/akanyaani/gpt-2-tensorflow2.0/blob/master/layers/embedding_layer.py
    - https://arxiv.org/pdf/1702.01417.pdf
"""


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, embeddings_initializer='uniform',
                 name='TokenAndPositionEmbedding'):
        super(TokenAndPositionEmbedding, self).__init__(name=name)
        self.maxlen, self.vocab_size, self.embed_dim = maxlen, vocab_size, embed_dim
        self.embeddings_initializer = embeddings_initializer
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                                   embeddings_initializer=embeddings_initializer,
                                                   name='SymbolEmbedding')
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim,
                                                 embeddings_initializer=embeddings_initializer,
                                                 name='PositionEmbedding')

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = {
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embeddings_initializer':
                tf.keras.initializers.serialize(tf.keras.initializers.get(self.embeddings_initializer)),
            'embed_dim': self.embed_dim,
        }

        base_config = super(TokenAndPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Isotropic(tf.keras.regularizers.Regularizer):
    """
    inspired by https://arxiv.org/pdf/1702.01417.pdf
        encourage std of singular values to be zero
    """

    def __init__(self, coef=10.):  # pylint: disable=redefined-outer-name
        self.coef = K.cast_to_floatx(coef)

    def __call__(self, x):
        # encourage std of singular values to be zero
        s, _, _ = tf.linalg.svd(x)
        loss = tf.square(tf.math.reduce_std(s))
        return self.coef * loss

    def get_config(self):
        return {'coef': float(self.coef)}


class ZeroMeanEmbedding(tf.keras.layers.Embedding):
    """
    inspired by https://arxiv.org/pdf/1702.01417.pdf
        embeddings with mean zero work best
    """

    def call(self, inputs):
        dtype = tf.keras.backend.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')

        if isinstance(self.embeddings, sharded_variable.ShardedVariable):
            emb = self.embeddings.variables
        else:
            emb = self.embeddings

        mean_embedding = tf.reduce_mean(emb, axis=0)[None]
        out = embedding_ops.embedding_lookup_v2(emb - mean_embedding, inputs)

        return out


class SymbolAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, embeddings_initializer='orthogonal',
                 name='SymbolAndPositionEmbedding',
                 symbol_embedding='zero_mean',
                 position_embedding=None, factorized_dim=None):
        super(SymbolAndPositionEmbedding, self).__init__(name=name)

        self.maxlen, self.vocab_size, self.embed_dim = maxlen, vocab_size, embed_dim
        self.embeddings_initializer = embeddings_initializer
        self.symbol_embedding = symbol_embedding
        self.position_embedding = position_embedding
        self.factorized_dim = factorized_dim

        if not factorized_dim is None:
            self.fs = tf.keras.layers.Dense(embed_dim)
            if not position_embedding == 'None':
                self.fp = tf.keras.layers.Dense(embed_dim)
            else:
                self.fp = lambda x: x
            embed_dim = factorized_dim
        else:
            self.fs = lambda x: x
            self.fp = lambda x: x

        symbol_regularizer = Isotropic() if 'isotropic' in symbol_embedding else None
        if 'learned' in symbol_embedding:
            self.sym_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                                     embeddings_initializer=embeddings_initializer,
                                                     embeddings_regularizer=symbol_regularizer,
                                                     name='SymbolEmbedding')

        elif 'onehot' in symbol_embedding:
            self.sym_emb = OneHot(vocab_size, name='SymbolEmbedding')

        elif 'zero_mean' in symbol_embedding:
            self.sym_emb = ZeroMeanEmbedding(input_dim=vocab_size, output_dim=embed_dim,
                                             embeddings_initializer=embeddings_initializer,
                                             embeddings_regularizer=symbol_regularizer,
                                             name='SymbolEmbedding')

        else:
            self.sym_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                                     embeddings_initializer=embeddings_initializer,
                                                     embeddings_regularizer=symbol_regularizer,
                                                     name='SymbolEmbedding')

        position_regularizer = Isotropic() if 'isotropic' in position_embedding else None
        if 'learned' in position_embedding:
            self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim,
                                                     embeddings_initializer=embeddings_initializer,
                                                     embeddings_regularizer=position_regularizer,
                                                     name='PositionEmbedding')

        elif 'zero_mean' in position_embedding:
            self.pos_emb = ZeroMeanEmbedding(input_dim=maxlen, output_dim=embed_dim,
                                             embeddings_initializer=embeddings_initializer,
                                             embeddings_regularizer=position_regularizer,
                                             name='PositionEmbedding')

        elif 'sinusoid' in position_embedding:
            self.pos_emb = lambda x: get_position_sinusoid(self.maxlen, embed_dim)

        else:
            self.pos_emb = lambda x: 0

    def embedding(self, inputs):
        with tf.name_scope("embedding"):
            seq_len = tf.shape(inputs)[1]

            positions = tf.range(start=0, limit=seq_len, delta=1)
            positions = self.fp(self.pos_emb(positions))
            x = self.fs(self.sym_emb(inputs))

            return x + positions

    def projection(self, inputs):
        with tf.name_scope('output_layer'):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            print('here!')
            # print(self.sym_emb.embeddings.variables)
            print(self.sym_emb.embeddings.shape)
            logits = tf.matmul(inputs, self.fs(self.sym_emb.embeddings),
                               transpose_b=True)

            return tf.reshape(logits, [batch_size, seq_len, self.vocab_size])

    def call(self, inputs, mode='embedding'):

        if mode == 'embedding':
            apply = self.embedding
        elif mode == 'projection':
            apply = self.projection
        else:
            raise ValueError("mode {} is not valid.".format(mode))

        return apply(inputs)

    # def compute_output_shape(self, input_shape)

    def get_config(self):
        config = {
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embeddings_initializer':
                tf.keras.initializers.serialize(tf.keras.initializers.get(self.embeddings_initializer)),
            'embed_dim': self.embed_dim,
            'symbol_embedding': self.symbol_embedding,
            'position_embedding': self.position_embedding,
            'factorized_dim': self.factorized_dim
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EmbeddingLayer(tf.keras.layers.Layer):
    """
    originally: https://github.com/akanyaani/gpt-2-tensorflow2.0/blob/master/layers/embedding_layer.py
    """

    def __init__(self, vocab_size, embedding_size, initializer=None, stddev=0.01, mean=0.0, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.stddev = stddev
        self.mean = mean
        self.initializer = initializer
        if self.initializer is None:
            self.initializer = tf.random_normal_initializer(mean=self.mean,
                                                            stddev=self.stddev)

    def build(self, input_shape):
        self.embedding_weights = self.add_weight(
            "weights",
            shape=[self.vocab_size, self.embedding_size],
            dtype="float32",
            initializer=self.initializer
        )
        super(EmbeddingLayer, self).build(input_shape)

    def call(self, inputs, mode="embedding", scale=False):
        if mode == "embedding":
            return self.embedding(inputs, scale=scale)
        elif mode == "projection":
            return self.projection(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def embedding(self, inputs, scale=False):
        with tf.name_scope("embedding"):
            # Create binary mask of size [batch_size, length]
            mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
            inputs = tf.cast(inputs, tf.int32)
            embeddings = tf.nn.embedding_lookup(self.embedding_weights, inputs)
            embeddings *= tf.expand_dims(mask, -1)
            # Scale embedding by the sqrt of the hidden size
            if scale:
                embeddings *= self.embedding_size ** 0.5

            return embeddings

    def projection(self, inputs):
        with tf.name_scope("output_layer"):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]

            h_flat = tf.reshape(inputs, [-1, self.embedding_size])
            logits = tf.matmul(h_flat, self.embedding_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, seq_len, self.vocab_size])


class PositionEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, position_seq, pos_embedding_size, trainable=True, stddev=0.02, mean=0.0, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.position_seq = position_seq
        self.hidden_size = pos_embedding_size
        self.trainable = trainable
        self.stddev = stddev
        self.mean = mean

        if trainable:
            self.position_embedding = EmbeddingLayer(self.position_seq, self.hidden_size,
                                                     stddev=self.stddev, mean=self.mean)

    def call(self, inputs, start=1):
        if self.trainable:
            batch_size = tf.shape(inputs)[0]
            batch_seq = tf.shape(inputs)[1]

            positions = tf.reshape(tf.tile(tf.range(start, batch_seq + start), [batch_size]),
                                   [batch_size, batch_seq])

            positions = tf.cast(positions, tf.int32)
            position_mask = tf.cast(tf.not_equal(inputs, 0), tf.int32)
            positions *= position_mask

            return self.position_embedding(positions)
        else:
            return get_position_sinusoid(self.position_seq)


# @staticmethod
def get_position_sinusoid(seq_len, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.cast(tf.range(seq_len), tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            tf.math.log(float(max_timescale) / float(min_timescale)) /
            (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal
