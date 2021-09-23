import tensorflow as tf
from transformers.generation_tf_utils import TFGenerationMixin

from GenericTools.KerasTools.esoteric_models.wizard_of_wikipedia import EndToEndModel
from GenericTools.LeanguageTreatmentTools.random_language import random_indices


class HF_outputs:
    def __init__(self, logits):
        self.logits = logits

    def __len__(self):
        return 1


class HF_ModelUpgrade(TFGenerationMixin):
    def __init__(self, model, encoder_inputs, bos_token_id, eos_token_id, pad_token_id, vocab_size):
        self.model = model
        self.encoder_inputs = encoder_inputs
        self.config = lambda x: x
        self.config.do_sample = False
        self.config.early_stopping = False
        self.config.num_beams = 1
        self.config.temperature = 1
        self.config.top_k = 100
        self.config.top_p = 1.
        self.config.repetition_penalty = 1.2
        self.config.bos_token_id = bos_token_id
        self.config.eos_token_id = eos_token_id
        self.config.pad_token_id = pad_token_id
        self.config.length_penalty = 1
        self.config.no_repeat_ngram_size = 0
        self.config.bad_words_ids = None
        self.config.num_return_sequences = 1
        self.config.decoder_start_token_id = bos_token_id
        self.config.forced_bos_token_id = None
        self.config.forced_eos_token_id = None
        self.config.vocab_size = vocab_size
        self.config.is_encoder_decoder = False
        self.config.output_scores = False
        self.config.return_dict_in_generate = False
        self.config.output_attentions = False
        self.config.output_hidden_states = False
        self.fixed_length_input = False
        self.curDim = 0

    def __call__(self, input_ids, return_dict=False, output_attentions=False, output_hidden_states=False):
        assert self.curDim < self.max_length
        self.curDim += 1
        repeats = self.num_beams * self.num_return_sequences if self.do_sample else self.num_beams
        encoder_inputs = [
            tf.repeat(t, repeats=repeats, axis=0)
            for t in self.encoder_inputs
        ]

        # print(encoder_inputs[0].shape, input_ids.shape, self.do_sample)
        if self.fixed_length_input:
            b = input_ids.shape[0]
            start_ids = tf.cast(self.config.bos_token_id * tf.ones((b, self.max_length - self.curDim)), input_ids.dtype)
            print(input_ids)
            print(start_ids)
            start_ids = tf.concat([input_ids, start_ids], axis=1)
            input_ids = tf.cast(start_ids, tf.int32)
        prediction = self.model(encoder_inputs + [input_ids])

        if self.fixed_length_input:
            prediction = prediction[:, :self.curDim]

        return HF_outputs(prediction)

    def get_output_embeddings(self):
        return 0

    def generate(self, fixed_length_input=False, **kwargs):
        self.curDim = 0
        self.fixed_length_input = fixed_length_input
        self.__dict__.update(kwargs)
        self.__dict__.update({'fixed_length_input': fixed_length_input})
        return super().generate(**kwargs)


def huggingface_upgrade():
    import numpy as np
    np.random.seed(2)
    max_knowledge = 5
    vocab_size = int(5e4)
    pad_idx = 3
    # tf.compat.v1.disable_eager_execution()
    model = EndToEndModel(max_knowledge=max_knowledge, input_vocab_size=vocab_size, pad_idx=pad_idx)

    batch_size = 5
    maxlen = 10
    src_tokens = random_indices(vocab_size, pad_idx=pad_idx, batch_size=batch_size, maxlen=4)
    tgt_tokens = random_indices(vocab_size, pad_idx=pad_idx, maxlen=maxlen, batch_size=batch_size)
    know_tokens = tf.concat([random_indices(vocab_size, pad_idx=pad_idx, batch_size=batch_size, maxlen=9)[:, None]
                             for _ in range(max_knowledge)], axis=1)
    chosen_knowledge = random_indices(max_knowledge, maxlen=1, batch_size=batch_size)
    input_tensors = [src_tokens, know_tokens, chosen_knowledge, tgt_tokens]

    # output_1 = model(input_tensors)
    hf_model = HF_ModelUpgrade(model, input_tensors[:-1], pad_idx, pad_idx, pad_idx, vocab_size)

    outputs = hf_model.generate(
        input_ids=tf.constant(tgt_tokens[:, 0][..., None]), num_beams=3, num_return_sequences=2, do_sample=False,
        max_length=maxlen, min_length=3, fixed_length_input=False
    )

    for i in range(3):
        print('\nGenerated {}: {}'.format(i, outputs[i]))

    outputs = hf_model.generate(
        input_ids=tf.constant(tgt_tokens[:, 0][..., None]), num_beams=3, num_return_sequences=2, do_sample=False,
        max_length=maxlen, min_length=3, fixed_length_input=True
    )

    for i in range(3):
        print('\nGenerated {}: {}'.format(i, outputs[i]))


if __name__ == '__main__':
    huggingface_upgrade()
