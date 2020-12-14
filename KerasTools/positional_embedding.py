import tensorflow as tf

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, embeddings_initializer='uniform',
                 name='TokenAndPositionEmbedding'):
        super(TokenAndPositionEmbedding, self).__init__(name=name)
        self.maxlen, self.vocab_size, self.embed_dim = maxlen, vocab_size, embed_dim
        self.embeddings_initializer = embeddings_initializer
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                                   embeddings_initializer=embeddings_initializer)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim,
                                                 embeddings_initializer=embeddings_initializer)

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
                tf.keras.initializers.serialize(self.embeddings_initializer),
            'embed_dim': self.embed_dim,
        }

        base_config = super(TokenAndPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
