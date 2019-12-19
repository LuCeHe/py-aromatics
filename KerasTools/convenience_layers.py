import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Embedding, \
    LSTM, Lambda, Dense, Layer
from tensorflow.keras.models import Model

from GenericTools.TFTools.convenience_operations import slice_from_to, clip_layer, replace_column


class ExpandDims(object):

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, inputs):
        def ed(tensor, axis):
            expanded = K.expand_dims(tensor, axis=axis)
            return expanded

        return Lambda(ed, arguments={'axis': self.axis})(inputs)

class Squeeze(object):

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, inputs):
        def squeeze(tensor, axis):
            squeezed = K.squeeze(tensor, axis=axis)
            return squeezed

        return Lambda(squeeze, arguments={'axis': self.axis})(inputs)

class Slice(object):

    # axis parameter is not functional
    def __init__(self, axis, initial, final):
        self.axis, self.initial, self.final = axis, initial, final

    def __call__(self, inputs):
        return Lambda(slice_from_to, arguments={'initial': self.initial, 'final': self.final})(inputs)


class Clip(object):

    def __init__(self, min_value=0., max_value=1.):
        self.min_value, self.max_value = min_value, max_value

    def __call_(self, inputs):
        return Lambda(clip_layer, arguments={'min_value': self.min_value, 'max_value': self.max_value})(inputs)


class ReplaceColumn(Layer):

    def __init__(self, column_position, **kwargs):
        super(ReplaceColumn, self).__init__(**kwargs)

        self.column_position = column_position

    def call(self, inputs, training=None):
        matrix, column = inputs

        matrix = tf.cast(matrix, dtype=tf.float32)
        column = tf.cast(column, dtype=tf.float32)
        new_matrix = replace_column(matrix, column, self.column_position)
        new_matrix = tf.cast(new_matrix, dtype=tf.int32)
        return new_matrix


def predefined_model(vocab_size, emb_dim, units=256):
    embedding = Embedding(vocab_size, emb_dim, mask_zero='True')
    lstm = LSTM(units, return_sequences=False)

    input_question = Input(shape=(None,), name='discrete_sequence')
    embed = embedding(input_question)
    lstm_output = lstm(embed)
    softmax = Dense(vocab_size, activation='softmax')(lstm_output)

    return Model(inputs=input_question, outputs=softmax)
