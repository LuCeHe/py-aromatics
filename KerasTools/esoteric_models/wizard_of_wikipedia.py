# I'm turning into tf the code for the article
# WIZARD OF WIKIPEDIA: KNOWLEDGE-POWERED CONVERSATIONAL AGENTS
# that was originally written in PyTorch in the ParlAI library
from tensorflow.keras.layers import *
import tensorflow as tf

tf.executing_eagerly()

from GenericTools.KerasTools.advanced_losses import sparse_perplexity, sparse_f1_on_max
from GenericTools.KerasTools.esoteric_models.transformer import TransformerEncoder, TransformerDecoder, \
    PaddingLookAheadMasks, create_padding_mask, create_look_ahead_mask
from GenericTools.LeanguageTreatmentTools.random_language import random_indices


def metrics_wow(num_classes):
    metrics = [
        sparse_perplexity,
        sparse_f1_on_max(num_classes),
    ]
    return metrics


def universal_sentence_embedding(sentences, mask, sqrt=True, epsilon=1e-6):
    """
    Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).

    This is really just sum / sqrt(len).

    :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings

    :return: an N x D matrix of sentence embeddings
    :rtype Tensor:
    """
    # need to mask out the padded chars
    sentence_sums = tf.einsum('bji,bj->bi', sentences, mask)

    divisor = tf.reduce_sum(mask, axis=1)
    if sqrt:
        divisor = tf.sqrt(divisor)

    sentence_sums /= divisor[..., None] + epsilon
    return sentence_sums


class UniversalSentenceEmbedding(tf.keras.layers.Layer):
    def __init__(self, invert_mask=True, **kwargs):
        self.invert_mask = invert_mask
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        sentences, mask = inputs
        if self.invert_mask:
            mask = 1 - mask
        sentence_sums = universal_sentence_embedding(sentences, mask, sqrt=True)
        return sentence_sums


class tf_ContextKnowledgeEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.transformer_encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, input_vocab_size,
                                                      maximum_position_encoding, rate)

    def call(self, inputs, *args, **kwargs):
        if isinstance(inputs, list):
            if len(inputs) == 3:
                src_tokens, know_tokens, chosen_knowledge = inputs
            elif len(inputs) == 2:
                src_tokens, know_tokens = inputs
                chosen_knowledge = None
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        context_mask = create_padding_mask(src_tokens, pad_idx=self.pad_idx)

        N = tf.shape(know_tokens)[0]
        K = tf.shape(know_tokens)[1]
        Tk = tf.shape(know_tokens)[2]

        know_flat = tf.reshape(know_tokens, (-1, Tk))
        knw_mask = create_padding_mask(know_flat, pad_idx=self.pad_idx)

        context_encoded = self.transformer_encoder(src_tokens, mask=context_mask)
        know_encoded = self.transformer_encoder(know_flat, mask=knw_mask)

        context_use = universal_sentence_embedding(context_encoded, tf.squeeze(1 - context_mask, [1, 2]), sqrt=True)
        know_use = universal_sentence_embedding(know_encoded, tf.squeeze(1 - knw_mask, [1, 2]), sqrt=True)

        know_use = tf.reshape(know_use, (N, K, self.d_model))

        context_use = context_use / tf.sqrt(tf.cast(self.d_model, tf.float32))
        know_use = know_use / tf.sqrt(tf.cast(self.d_model, tf.float32))

        ck_attn = tf.einsum('bij,bj->bi', know_use, context_use)

        if chosen_knowledge is None:
            chosen_knowledge = tf.argmax(ck_attn, axis=1)
            chosen_knowledge = tf.expand_dims(chosen_knowledge, 1)
        else:
            expandaded_ck_attn = tf.expand_dims(ck_attn, 1)

            loss = .1 * tf.keras.losses.SparseCategoricalCrossentropy()(chosen_knowledge, expandaded_ck_attn)
            self.add_loss(loss)
            self.add_metric(loss, name='knowledge_loss', aggregation='mean')

        koh = tf.squeeze(tf.one_hot(tf.cast(chosen_knowledge, tf.int32), K), 1)

        know_encoded = tf.reshape(know_encoded, (N, K, Tk, self.d_model))
        knw_mask = tf.reshape(knw_mask, (N, K, Tk))

        cs_encoded = tf.einsum('bijk,bi->bjk', know_encoded, koh)
        cs_mask = tf.einsum('bij,bi->bj', knw_mask, koh)

        full_enc = tf.concat([cs_encoded, context_encoded], axis=1)
        cs_mask = tf.expand_dims(tf.expand_dims(cs_mask, axis=1), axis=1)
        full_mask = tf.concat([cs_mask, context_mask], axis=3)

        return full_enc, full_mask, ck_attn  # (batch_size, input_seq_len, d_model)


class tf_ContextKnowledgeDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.transformer_decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, input_vocab_size,
                                                      maximum_position_encoding, rate)

    def call(self, inputs, output_type='embedding_projection', *args, **kwargs):
        tgt_tokens, encoder_state = inputs
        encoder_output, encoder_mask, _ = encoder_state

        look_ahead_mask = create_look_ahead_mask(tf.shape(tgt_tokens)[1])
        dec_target_padding_mask = create_padding_mask(tgt_tokens)
        decoder_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        output = self.transformer_decoder(
            tgt_tokens, encoder_output, decoder_mask, encoder_mask, output_type=output_type)[0]
        output = output - tf.reduce_max(output, axis=-1, keepdims=True)
        return output  # (batch_size, input_seq_len, d_model)


#
# original implementation hyperparams
#
# 'multitask_weights': [1], 'batchsize': 64, 'batch_sort': False,
# 'ignorant_dropout': 0.0, 'embedding_size': 256, 'n_layers': 5,
# 'ffn_size': 512, 'attention_dropout': 0.0, 'relu_dropout': 0.0,
# 'dropout': 0.2, 'n_heads': 2, 'learn_positional_embeddings': False,
# 'embeddings_scale': True, 'n_positions': 128, 'init_model': None,
# 'beam_size': 1, 'embedding_type': 'fasttext', 'embedding_projection': 'random',
# 'optimizer': 'adam', 'learningrate': 0.0005, 'gradient_clip': 0.1,
# 'momentum': 0, 'nesterov': True, 'nus': [0.7], 'betas': [0.9, 0.98],
# 'lr_scheduler': 'invsqrt', 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5,
# 'warmup_updates': 5000, 'warmup_rate': 0.0001, 'update_freq': -1,
# 'knowledge_truncate': 32, 'max_knowledge': 32, 'knowledge_alpha': 0.95,
# 'embeddingsize': 256, 'clip': 0.1, 'clip_norm': 0.1, 'activation': 'relu',
# 'output_scaling': 1.0, 'share_word_embeddings': True, 'n_encoder_layers': -1,
# 'n_decoder_layers': -1, 'beam_context_block_ngram': -1, 'beam_length_penalty': 0.65,
# 'topk': 10, 'topp': 0.9, 'beam_delay': 30, 'beam_block_list_filename': None,
# 'temperature': 1.0, 'adam_eps': 1e-08, 'adafactor_eps': (1e-30, 0.001), 'weight_decay': None,

def EndToEndModel(num_layers=5, d_model=256, num_heads=2, dff=512, input_vocab_size=int(5e4),
                  target_vocab_size=int(5e4), max_pos=1024, rate=.1, max_knowledge=5, pad_idx=0):
    cke = tf_ContextKnowledgeEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_pos, rate, pad_idx)
    ckd = tf_ContextKnowledgeDecoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_pos, rate, pad_idx)

    src_tokens = Input((None,))
    tgt_tokens = Input((None,))
    know_tokens = Input((max_knowledge, None))
    chosen_knowledge = Input((1,))

    code = cke([src_tokens, know_tokens, chosen_knowledge])
    logits = ckd([tgt_tokens, code], output_type='embedding_projection')

    model = tf.keras.models.Model([src_tokens, know_tokens, chosen_knowledge, tgt_tokens], logits)

    src_tokens = Input((None,))
    tgt_tokens = Input((None,))
    know_tokens = Input((max_knowledge, None))

    code = cke([src_tokens, know_tokens])
    logits = ckd([tgt_tokens, code], output_type='embedding_projection')[0]

    test_model = tf.keras.models.Model([src_tokens, know_tokens, tgt_tokens], logits)
    return model, test_model


def EndToEndModelGPT2(num_layers=5, d_model=256, num_heads=2, dff=512, input_vocab_size=int(5e4),
                      target_vocab_size=int(5e4), max_pos=1024, rate=.1, max_knowledge=5, pad_idx=0):
    cke = tf_ContextKnowledgeEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_pos, rate, pad_idx)
    ckd = tf_ContextKnowledgeDecoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_pos, rate, pad_idx)

    src_tokens = Input((None,))
    tgt_tokens = Input((None,))
    know_tokens = Input((max_knowledge, None))
    chosen_knowledge = Input((1,))

    code = cke([src_tokens, know_tokens, chosen_knowledge])
    logits = ckd([tgt_tokens, code], output_type='embedding_projection')

    model = tf.keras.models.Model([src_tokens, know_tokens, chosen_knowledge, tgt_tokens], logits)

    src_tokens = Input((None,))
    tgt_tokens = Input((None,))
    know_tokens = Input((max_knowledge, None))

    code = cke([src_tokens, know_tokens])
    logits = ckd([tgt_tokens, code], output_type='embedding_projection')[0]

    test_model = tf.keras.models.Model([src_tokens, know_tokens, tgt_tokens], logits)
    return model, test_model


def quick_test():
    max_knowledge = 5
    input_vocab_size = int(5e4)
    model, test_model = EndToEndModel(max_knowledge=max_knowledge, input_vocab_size=input_vocab_size)
    vocab_size = 20

    src_tokens = random_indices(vocab_size)
    tgt_tokens = random_indices(vocab_size)
    know_tokens = tf.concat([random_indices(vocab_size)[:, None] for _ in range(max_knowledge)], axis=1)
    chosen_knowledge = random_indices(max_knowledge, maxlen=1)
    input_tensors = [src_tokens, know_tokens, chosen_knowledge, tgt_tokens]
    input_test_tensors = [src_tokens, know_tokens, tgt_tokens]

    print(src_tokens.shape, know_tokens.shape, chosen_knowledge.shape, tgt_tokens.shape)
    output = model(input_tensors)
    print(output.shape)

    prediction = model.predict(input_tensors)
    print(prediction.shape)

    model.compile(
        'SGD', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics_wow(num_classes=input_vocab_size))
    model.fit(input_tensors, tgt_tokens, epochs=3)

    prediction = test_model.predict(input_test_tensors)
    print(prediction.shape)

    # test_model.compile(
    #     'SGD', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=metrics_wow(num_classes=input_vocab_size))
    # test_model.fit(input_test_tensors, tgt_tokens, epochs=3)


if __name__ == '__main__':
    quick_test()
