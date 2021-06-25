import tensorflow as tf
import os, shutil
from tokenizers import Tokenizer

from datasets import load_dataset as true_load_dataset
# from transformers import MarianTokenizer
import numpy as np

from GenericTools.KerasTools.esoteric_tasks.translation_load import load_dataset

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', 'translate'))
tokenizer_path = os.path.join(DATAPATH, 'wmt14_tokenizer.json')
destination_tokenizer = os.path.abspath(os.path.join(CDIR, '..', 'data_generators', 'wmt14_tokenizer.json'))


class Wmt14Generator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            maxlen=20,
            neutral_phase_length=1,
            repetitions=2,
            train_val_test='train',
            category_coding='onehot'):
        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            maxlen=maxlen,
            neutral_phases=neutral_phase_length,
            repetitions=repetitions,
            train_val_test=train_val_test,
            n_regularizations=0, )

        self.out_len = 100  # 390 - 1
        self.in_len = self.out_len * repetitions

        self.on_epoch_end()

        # def on_epoch_end(self):
        self.dataset = load_dataset("wmt14", 'de-en', cache_dir=DATAPATH,
                                    split=self.train_val_test.replace('val', 'validation'))
        # dataset = load_dataset("wmt14", 'de-en', cache_dir=DATAPATH, split='validation')
        self.tokenizer = Tokenizer.from_file(destination_tokenizer)

        self.lines = len(self.dataset)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.id_to_word = [k for k, _ in self.tokenizer.get_vocab().items()]
        # self.encoded_dataset = dataset.map(self.string2tokens, batched=True)

        if self.steps_per_epoch == None:
            self.steps_per_epoch = int(self.lines / self.batch_size)-1

        self.in_dim = 1
        self.out_dim = self.vocab_size

        self.epochs = 20 if epochs == None else epochs

    def on_epoch_end(self):
        self.batch_i = 0

    def decode(self, list_ids):
        decoded_sequence = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in list_ids]
        return decoded_sequence

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        batch = self.data_generation()
        return (batch['input_spikes'], batch['mask']), (batch['target_output'], *[np.array(0)] * self.n_regularizations)

    def data_generation(self):
        bi = self.batch_size * self.batch_i
        bf = self.batch_size * (self.batch_i + 1)

        sentence_pairs = [[self.dataset[i]['translation']['en'], self.dataset[i]['translation']['de']]
                          for i in range(bi, bf)]
        output = self.tokenizer.encode_batch(sentence_pairs)

        batch = np.array([o.ids for o in output])
        mask = np.array([o.attention_mask for o in output])
        # batch = batch[..., None]

        self.batch_i += 1

        input_batch = batch[:, :-1, None]
        input_batch = np.repeat(input_batch, self.repetitions, axis=1)

        output_batch = batch[:, 1:]
        output_batch = tf.keras.utils.to_categorical(output_batch, num_classes=self.vocab_size)

        mask = mask[:, 1:, None]
        mask = np.repeat(mask, self.vocab_size, axis=2)

        return {'input_spikes': input_batch, 'target_output': output_batch,
                'mask': mask}


def test():
    dataset = load_dataset("wmt14", 'de-en', cache_dir=DATAPATH, split='validation')
    tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
    tok = Tokenizer.from_file(tokenizer_path)
    tok.enable_padding(pad_id=3, pad_token="[PAD]")

    def string2tokens(examples):
        text = ['[enenen]' + ' ' + e['en'] + ' ' + '[dedede]' + ' ' + e['de']
                for e in examples['translation']]
        BatchEncoding = tok.prepare_seq2seq_batch(text, return_tensors='np')
        # encoded = tok.encode_batch(text)
        # data_dict['input_ids'])
        # mask = np.array(data_dict['attention_mask']
        print(BatchEncoding)
        return BatchEncoding

    encoded_dataset = dataset.map(string2tokens, batched=True)
    batch_size = 32
    input_batch = np.array(encoded_dataset[:batch_size]['input_ids'])
    mask = np.array(encoded_dataset[:batch_size]['attention_mask'])
    print(input_batch.shape)

    print('\n\n\n')

    tgt_text = [tok.decode(t, skip_special_tokens=False) for t in [*input_batch]]

    for t in tgt_text:
        print(t)

    print('\n\n\n')

    print(tok.source_lang)
    print(tok.sep_token)
    print(tok.__dir__())
    print(tok._convert_token_to_id)
    print(tok.vocab_size)
    print(len(dataset))
    # print(tok.get_vocab())
    # print(tok.decoder)


def generate_sentences(encoded_dataset):
    for sentence in encoded_dataset:
        yield sentence['sentence']


def string2tokens(examples):
    text = examples['translation']['en'] + ' ' + examples['translation']['de']
    return {'sentence': text}


def download():
    # d = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
    max_length = 100
    a = true_load_dataset("wmt14", 'de-en', cache_dir=DATAPATH, split='validation', save_infos=True)
    b = true_load_dataset("wmt14", 'de-en', cache_dir=DATAPATH, split='train')
    c = true_load_dataset("wmt14", 'de-en', cache_dir=DATAPATH, split='test')

    a = load_dataset("wmt14", 'de-en', cache_dir=DATAPATH, split='validation', save_infos=True)
    b = load_dataset("wmt14", 'de-en', cache_dir=DATAPATH, split='train')
    c = load_dataset("wmt14", 'de-en', cache_dir=DATAPATH, split='test')

    if not os.path.isfile(tokenizer_path):
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.normalizers import Lowercase, NFKC, Sequence
        from tokenizers.pre_tokenizers import ByteLevel, Digits, Punctuation, Whitespace
        from tokenizers import pre_tokenizers
        from tokenizers.processors import TemplateProcessing
        from tokenizers.trainers import BpeTrainer

        # First we create an empty Byte-Pair Encoding model (i.e. not trained model)
        tokenizer = Tokenizer(BPE())
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [Whitespace(), Punctuation(), ByteLevel(), Digits(individual_digits=True)])
        # tokenizer.pre_tokenizer = ByteLevel()

        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=1000,
                             special_tokens=["[PAD]", "[CLS]", "[EOT]", "[UNK]", "[MASK]", "[enenen]", "[dedede]"])

        tokenizer.post_processor = TemplateProcessing(
            single="[enenen] $A [EOT]",
            pair="[enenen] $A [dedede] $B:1 [EOT]:1",
            special_tokens=[
                ("[enenen]", 5),
                ("[dedede]", 6),
                ("[EOT]", 2),
            ],
        )

        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        tokenizer.enable_truncation(max_length + 1)

        encoded_dataset = a.map(string2tokens, batched=False)
        tokenizer.train_from_iterator(generate_sentences(encoded_dataset), length=len(encoded_dataset), trainer=trainer)
        tokenizer.save(tokenizer_path)
        shutil.copy(tokenizer_path, destination_tokenizer)
    tokenizer = Tokenizer.from_file(destination_tokenizer)

    batch_i = 4
    batch_size = 3
    bi = batch_size * batch_i
    bf = batch_size * (batch_i + 1)

    sentence_pairs = [[a[i]['translation']['en'], a[i]['translation']['de']] for i in range(bi, bf)]
    output = tokenizer.encode_batch(sentence_pairs)
    # decoded_sequence = tokenizer.decode(output)
    print(output)
    print([o.tokens for o in output])
    print([o.ids for o in output])
    print([o.attention_mask for o in output])
    print([o.type_ids for o in output])
    print()
    decoded_sequence = [tokenizer.decode(o.ids, skip_special_tokens=False) for o in output]
    # decoded_sequence = tokenizer.decode_batch(output)
    print(decoded_sequence)
    print(tokenizer.decode([0, 532, 0]))
    print(tokenizer.get_vocab())
    print('DONE!')


if __name__ == '__main__':
    download()
    # test()
