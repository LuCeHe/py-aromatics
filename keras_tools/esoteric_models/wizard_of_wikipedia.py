# tensorflow version of the code for the article
# WIZARD OF WIKIPEDIA: KNOWLEDGE-POWERED CONVERSATIONAL AGENTS
# that was originally written in PyTorch in the ParlAI library
# https://gitlab.ilabt.imec.be/ahadifar/google-research/-/blob/16a8f847718f1ad824eb16680caabb7a79ae8411/dialogue_ope/airdialogue_model_transformer/models/modules.py
# https://github.com/facebookresearch/ParlAI/tree/main/projects/wizard_of_wikipedia

# from parlai.agents.transformer.modules import TransformerEncoder as pt_TransformerEncoder
# from projects.wizard_of_wikipedia.generator.modules import ContextKnowledgeDecoder as pt_ContextKnowledgeDecoder
# from projects.wizard_of_wikipedia.generator.modules import ContextKnowledgeEncoder as pt_ContextKnowledgeEncoder
from tensorflow.keras.layers import *
import tensorflow as tf

from pyaromatics.keras_tools.esoteric_layers import ContrastiveLossLayer, AddLossLayer, AddMetricsLayer
from pyaromatics.keras_tools.esoteric_losses import smape_loss
from pyaromatics.keras_tools.esoteric_losses.advanced_losses import *
from pyaromatics.keras_tools.esoteric_layers.random_switch import RandomSwitch
from pyaromatics.keras_tools.esoteric_models.transformer import TransformerEncoder as tf_TransformerEncoder, GPT
from pyaromatics.keras_tools.esoteric_models.transformer import TransformerDecoder as tf_TransformerDecoder
from pyaromatics.keras_tools.esoteric_models.transformer import create_padding_mask, create_look_ahead_mask


def metrics_wow(num_classes, mask_value):
    metrics = [
        sparse_perplexity,
        sparse_f1_on_max,
        # masked_sparse_crossentropy(mask_value),
        masked_sparse_perplexity(mask_value),
        masked_f1_on_max(num_classes, mask_value),
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

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, invert_mask=True, **kwargs):
        self.init_args = dict(invert_mask=invert_mask)
        self.__dict__.update(self.init_args)

        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        sentences, mask = inputs
        if self.invert_mask:
            mask = 1 - mask
        sentence_sums = universal_sentence_embedding(sentences, mask, sqrt=True)
        return sentence_sums


class KnowledgeDropout(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_switch = RandomSwitch([.8, .1, .1])

    def call(self, inputs, *args, **kwargs):
        koh = inputs

        look_all_knowledge = tf.ones_like(koh) / tf.cast(tf.shape(koh)[-1], tf.float32)
        look_no_knowledge = tf.zeros_like(koh)

        switched = self.random_switch([koh, look_all_knowledge, look_no_knowledge])
        lp = tf.cast(tf.keras.backend.learning_phase(), tf.float32)

        switched = lp * switched + (1 - lp) * koh
        return switched


class tf_ContextKnowledgeEncoder(tf.keras.layers.Layer):

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding=1024, rate=0.1, pad_idx=0, config='original_transformer', **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                              input_vocab_size=input_vocab_size, maximum_position_encoding=maximum_position_encoding,
                              rate=rate, pad_idx=pad_idx, config=config)
        self.__dict__.update(self.init_args)

        self.transformer_encoder = tf_TransformerEncoder(num_layers, d_model, num_heads, dff, input_vocab_size,
                                                         maximum_position_encoding, rate, config=config)
        self.knowledge_dropout = KnowledgeDropout()

    def build(self, input_shape):
        self.use_external_knowledge = self.add_weight(
            name='use_external_knowledge', shape=(), initializer=tf.keras.initializers.Constant(1.), trainable=False
        )
        self.built = True

    def call(self, inputs, *args, **kwargs):
        src_tokens, know_tokens, chosen_knowledge = inputs

        context_mask = create_padding_mask(src_tokens, pad_idx=self.pad_idx)

        batch_size = tf.shape(know_tokens)[0]
        K = tf.shape(know_tokens)[1]
        Tk = tf.shape(know_tokens)[2]

        know_flat = tf.reshape(know_tokens, (-1, Tk))
        knw_mask = create_padding_mask(know_flat, pad_idx=self.pad_idx)

        context_encoded = self.transformer_encoder(src_tokens, mask=context_mask)
        know_encoded = self.transformer_encoder(know_flat, mask=knw_mask)

        context_use = universal_sentence_embedding(context_encoded, tf.squeeze(1 - context_mask, [1, 2]), sqrt=True)
        know_use = universal_sentence_embedding(know_encoded, tf.squeeze(1 - knw_mask, [1, 2]), sqrt=True)

        know_use = tf.reshape(know_use, (batch_size, K, self.d_model))

        context_use = context_use / tf.sqrt(tf.cast(self.d_model, tf.float32))
        know_use = know_use / tf.sqrt(tf.cast(self.d_model, tf.float32))

        ck_attn = tf.einsum('bij,bj->bi', know_use, context_use)

        network_knowledge_choice = tf.argmax(ck_attn, axis=1)
        network_knowledge_choice = tf.cast(tf.expand_dims(network_knowledge_choice, 1), tf.float32)

        knowledge = self.use_external_knowledge * chosen_knowledge \
                    + (1 - self.use_external_knowledge) * network_knowledge_choice

        expandaded_ck_attn = tf.expand_dims(ck_attn, 1)

        loss = 1. * tf.keras.losses.SparseCategoricalCrossentropy()(chosen_knowledge, expandaded_ck_attn)
        self.add_loss(loss)
        self.add_metric(loss, name='knowledge_loss', aggregation='mean')

        koh = tf.squeeze(tf.one_hot(tf.cast(knowledge, tf.int32), K), 1)

        # FIXME: knowledge_dropout will cause probs with cs_mask
        koh = self.knowledge_dropout(koh)
        know_encoded = tf.reshape(know_encoded, (batch_size, K, Tk, self.d_model))
        knw_mask = tf.reshape(knw_mask, (batch_size, K, Tk))

        cs_encoded = tf.einsum('bijk,bi->bjk', know_encoded, koh)
        cs_mask = tf.einsum('bij,bi->bj', knw_mask, koh)

        full_enc = tf.concat([cs_encoded, context_encoded], axis=1)
        cs_mask = tf.expand_dims(tf.expand_dims(cs_mask, axis=1), axis=1)
        full_mask = tf.concat([cs_mask, context_mask], axis=3)

        return full_enc, full_mask, ck_attn  # (batch_size, input_seq_len, d_model)


class tf_ContextKnowledgeDecoder(tf.keras.layers.Layer):

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, pad_idx=0, config='original_transformer', **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                              input_vocab_size=input_vocab_size, maximum_position_encoding=maximum_position_encoding,
                              rate=rate, pad_idx=pad_idx, config=config)
        self.__dict__.update(self.init_args)

        self.transformer_decoder = tf_TransformerDecoder(num_layers, d_model, num_heads, dff, input_vocab_size,
                                                         maximum_position_encoding, rate, config=config)

    def call(self, inputs, output_type='embedding_projection', *args, **kwargs):
        tgt_tokens, encoder_state = inputs
        encoder_output, encoder_mask, _ = encoder_state

        look_ahead_mask = create_look_ahead_mask(tf.shape(tgt_tokens)[1])
        dec_target_padding_mask = create_padding_mask(tgt_tokens, pad_idx=self.pad_idx)
        decoder_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        output = self.transformer_decoder(
            tgt_tokens, enc_output=encoder_output, look_ahead_mask=decoder_mask, padding_mask=encoder_mask,
            output_type=output_type)[0]
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
                  target_vocab_size=int(5e4), encoder_maxlen=1024, decoder_maxlen=1024,
                  rate=.1, max_knowledge=5, pad_idx=0, comments='original_transformer'):
    cke = tf_ContextKnowledgeEncoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                     input_vocab_size=input_vocab_size, maximum_position_encoding=encoder_maxlen,
                                     rate=rate, pad_idx=pad_idx, config=comments)
    ckd = tf_ContextKnowledgeDecoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                     input_vocab_size=input_vocab_size, maximum_position_encoding=decoder_maxlen,
                                     rate=rate, pad_idx=pad_idx, config=comments)

    src_tokens = Input((None,))
    tgt_tokens = Input((None,))
    know_tokens = Input((max_knowledge, None))
    chosen_knowledge = Input((1,))
    output_tokens = Input((None,))

    code = cke([src_tokens, know_tokens, chosen_knowledge])
    logits = ckd([tgt_tokens, code], output_type='embedding_projection')

    if 'contrastive' in comments:
        c = ContrastiveLossLayer(string_config=comments)
        logits = c([output_tokens, logits])

    # logits = AddLossLayer(loss=sparse_perplexity)([output_tokens, logits])
    logits = AddLossLayer(loss=sparsesmape)([output_tokens, logits])
    logits = AddMetricsLayer(metrics=metrics_wow(num_classes=input_vocab_size, mask_value=pad_idx))(
        [output_tokens, logits])

    model = tf.keras.models.Model([src_tokens, know_tokens, chosen_knowledge, tgt_tokens, output_tokens], logits)
    return model


def EndToEndModel_noKnowledge(num_layers=5, d_model=256, num_heads=2, dff=512, input_vocab_size=int(5e4),
                              target_vocab_size=int(5e4), encoder_maxlen=1024, decoder_maxlen=1024,
                              rate=.1, max_knowledge=5, pad_idx=0, datapath=''):
    ckd = GPT(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, target_vocab_size=input_vocab_size,
              maximum_position_encoding=decoder_maxlen, pad_idx=pad_idx, rate=rate)
    # ckd = tf_ContextKnowledgeDecoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
    #                                  input_vocab_size=input_vocab_size, maximum_position_encoding=decoder_maxlen,
    #                                  rate=rate, pad_idx=pad_idx)

    src_tokens = Input((None,))
    tgt_tokens = Input((None,))
    know_tokens = Input((max_knowledge, None))
    chosen_knowledge = Input((1,))
    output_tokens = Input((None,))

    logits = ckd(tgt_tokens, output_type='embedding_projection')
    # logits = AddLossLayer(loss=sparse_perplexity)([output_tokens, logits])
    logits = AddLossLayer(loss=sparsesmape)([output_tokens, logits])
    logits = AddMetricsLayer(metrics=metrics_wow(num_classes=input_vocab_size, mask_value=pad_idx))(
        [output_tokens, logits])

    model = tf.keras.models.Model([src_tokens, know_tokens, chosen_knowledge, tgt_tokens, output_tokens], logits)
    return model


def switch_external_knowledge(model, state='on'):
    if state == 'on':
        switch = 1.
    else:
        switch = 0.

    knowledge_switch = 'use_external_knowledge'
    for layer in model.layers:
        for weight in layer.weights:
            if knowledge_switch in weight.name:
                knowledge_weight = weight
    tf.keras.backend.set_value(knowledge_weight, switch)


def quick_test():
    import numpy as np
    np.random.seed(2)
    max_knowledge = 5
    vocab_size = int(5e4)
    pad_idx = 3

    model = EndToEndModel(max_knowledge=max_knowledge, input_vocab_size=vocab_size, pad_idx=pad_idx)

    batch_size = 5
    maxlen = 3
    src_tokens = random_indices(vocab_size, pad_idx=pad_idx, batch_size=batch_size, maxlen=4)
    tgt_tokens = random_indices(vocab_size, pad_idx=pad_idx, maxlen=maxlen, batch_size=batch_size)
    know_tokens = tf.concat([random_indices(vocab_size, pad_idx=pad_idx, batch_size=batch_size, maxlen=9)[:, None]
                             for _ in range(max_knowledge)], axis=1)

    chosen_knowledge = random_indices(max_knowledge, maxlen=1, batch_size=batch_size)
    input_tensors = [src_tokens, know_tokens, chosen_knowledge, tgt_tokens]

    print([t.shape for t in input_tensors])

    output_1 = model(input_tensors)
    output_2 = model(input_tensors)
    print('Is the reply of the network consistent with itself? ', np.all(output_2 == output_1))

    switch_external_knowledge(model, state='off')
    # output_3 = test_model(input_test_tensors)
    output_3 = model(input_tensors)
    print('Is the reply of the test network consistent with train network? ', np.all(output_2 == output_3))

    print(output_1.shape, output_3.shape)

    print('train and test model predictions')
    switch_external_knowledge(model, state='on')
    prediction = model.predict(input_tensors, steps=1)
    print(prediction.shape)

    switch_external_knowledge(model, state='off')
    prediction = model.predict(input_tensors, steps=1)
    print(prediction.shape)

    print('train and test model fit')
    switch_external_knowledge(model, state='on')
    model.compile(
        'SGD', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics_wow(num_classes=vocab_size, mask_value=pad_idx))
    model.fit(input_tensors, tgt_tokens, epochs=2, steps_per_epoch=1)

    switch_external_knowledge(model, state='off')
    model.compile(
        'SGD', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics_wow(num_classes=vocab_size, mask_value=pad_idx))
    model.fit(input_tensors, tgt_tokens, epochs=2, steps_per_epoch=1)


def long_test():
    from pyaromatics.LeanguageTreatmentTools.random_language import random_indices
    from pyaromatics.keras_tools.esoteric_optimizers.optimizer_selection import get_optimizer
    from pyaromatics.keras_tools.esoteric_tasks.wizard_of_wikipedia import WikipediaWizardGenerator

    max_knowledge = 32
    batch_size = 8
    n_dialogues = 'full'  # -1 100 random
    epochs = 1
    steps_per_epoch = None
    seed = 4
    vocab_size = int(3e4)
    tokenizer_choice = 'bpe'
    rate = .1

    encoder_maxlen = 128
    decoder_maxlen = 16

    gen_train = WikipediaWizardGenerator(
        data_path=DATAPATH, n_dialogues=n_dialogues, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
        encoder_maxlen=encoder_maxlen, decoder_maxlen=decoder_maxlen, epochs=epochs, tokenizer_choice=tokenizer_choice,
        data_split='train')
    gen_val = WikipediaWizardGenerator(
        data_path=DATAPATH, n_dialogues=n_dialogues, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
        encoder_maxlen=encoder_maxlen, decoder_maxlen=decoder_maxlen, tokenizer_choice=tokenizer_choice,
        data_split='valid_random_split')
    pad_idx = gen_train.pad_idx

    model = EndToEndModel(
        max_knowledge=max_knowledge, input_vocab_size=vocab_size, target_vocab_size=vocab_size, pad_idx=pad_idx,
        encoder_maxlen=encoder_maxlen, decoder_maxlen=decoder_maxlen, rate=rate)
    model.summary()
    optimizer = get_optimizer(
        'Adam', .0005, lr_schedule='cosine_no_restarts',
        total_steps=gen_train.epochs, clipnorm=.1,
        warmup_steps=gen_train.steps_per_epoch
    )
    model.compile(
        optimizer,
        masked_sparse_crossentropy(mask_value=pad_idx),
        metrics=metrics_wow(num_classes=vocab_size, mask_value=pad_idx)
    )
    model.fit(gen_train, epochs=gen_train.epochs, validation_data=gen_val)


if __name__ == '__main__':
    quick_test()
