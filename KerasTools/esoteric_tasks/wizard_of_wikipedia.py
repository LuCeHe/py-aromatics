import os, shutil, json, random, h5py, argparse
from tokenizers import Tokenizer
from transformers import GPT2Tokenizer, AutoTokenizer, AutoConfig
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import numpy as np
import pandas as pd

from GenericTools.LeanguageTreatmentTools.unpadding import unpad_sequence
from GenericTools.PlotTools.mpl_tools import load_plot_settings
from GenericTools.StayOrganizedTools.download_utils import download_and_unzip
from GenericTools.StayOrganizedTools.utils import str2val

pd = load_plot_settings(pd=pd)

split_names = ['valid_random_split', 'valid_topic_split', 'test_random_split', 'test_topic_split', 'train']


# split_names = ['train', 'valid_random_split', 'test_random_split']

def tokenize(sentence, tokenizer, tokenizer_choice):
    if tokenizer_choice == 'bpe':
        ids = tokenizer.encode(sentence).ids
    elif tokenizer_choice == 'gpt2':
        ids = tokenizer(sentence)['input_ids']
    else:
        raise NotImplementedError
    return ids


def download(data_path, tokenizer_choice, n_dialogues):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    safe_max_len = 256 if 'full' in n_dialogues else 512
    n_utterances_back = 4
    max_knowledge = 32
    reduce_data_by =2
    DATAPATH = data_path
    DATADESTINATION = os.path.join(DATAPATH, tokenizer_choice)
    os.makedirs(DATADESTINATION, exist_ok=True)

    random_or_full = str(n_dialogues)

    if 'random' in n_dialogues:
        n_dialogues = str2val(n_dialogues, 'random', int, default=None)
        predict_wizards = lambda x: [np.random.choice(x)]
    elif 'full' in n_dialogues:
        n_dialogues = str2val(n_dialogues, 'full', int, default=None)
        predict_wizards = lambda x: range(x)
    else:
        raise NotImplementedError

    assert tokenizer_choice in ['bpe', 'gpt2']
    tokenizer_path = os.path.join(DATADESTINATION, 'tokenizer-{}.json'.format(tokenizer_choice))

    if len(os.listdir(DATAPATH)) == 0:
        url = 'http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz'
        download_and_unzip([url], DATAPATH)

    if not os.path.isfile(tokenizer_path) and tokenizer_choice == 'bpe':
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace, Sequence, Digits

        url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip'
        download_and_unzip([url], DATAPATH)

        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))

        trainer = BpeTrainer(special_tokens=['[UNK]', '[PAD]', '[START]'])
        pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True)])
        tokenizer.pre_tokenizer = pre_tokenizer

        files = [os.path.join(DATAPATH, f'wikitext-103-raw/wiki.{split}.raw') for split in ['test', 'train', 'valid']]
        tokenizer.train(files, trainer)

        tokenizer.save(tokenizer_path)

        shutil.rmtree(os.path.join(DATAPATH, 'wikitext-103-raw'))

    elif not os.path.isfile(os.path.join(DATADESTINATION, 'config.json')) and tokenizer_choice == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.save_vocabulary(DATADESTINATION)

        config = AutoConfig.from_pretrained('gpt2')
        config.save_pretrained(DATADESTINATION)
    else:
        if tokenizer_choice == 'bpe':
            tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(DATADESTINATION)

    pad_word = '[PAD]' if tokenizer_choice == 'bpe' else '<|endoftext|>'
    pad_idx = tokenize(pad_word, tokenizer, tokenizer_choice)[0]

    large_df = pd.DataFrame(
        columns=['data_split', 'max_target_length', 'max_context_length', 'max_knowledge_length',
                 'max_knowledge_items', 'n_samples', 'n_dialogues', 'chosen_not_found'])

    for split_name in split_names:
        max_target_length, max_context_length, max_knowledge_length, max_knowledge_items = 0, 0, 0, 0
        chosen_not_found = 0

        h5_path = os.path.join(DATADESTINATION, '{}_{}_{}.h5'.format(split_name, tokenizer_choice, random_or_full))
        data_json = os.path.join(DATAPATH, split_name + '.json')

        if not os.path.isfile(h5_path):
            print(split_name)
            with open(data_json) as f:
                data = json.load(f)

            targets = []
            contexts = []
            knowledges = []
            choices = []
            for dialogue_i in tqdm(range(len(data))[:n_dialogues]):
                # print('-' * 59, dialogue_i)
                context = data[dialogue_i]['chosen_topic']
                target = ''
                wizard_count = 0
                chosen_topic_passage = [
                    data[dialogue_i]['chosen_topic'] + ': ' + ' '.join(data[dialogue_i]['chosen_topic_passage'])]
                zero_knowledge = ['[PAD][PAD]'] * max_knowledge
                dialogue_knowledges = [zero_knowledge] * n_utterances_back

                speakers = [d['speaker'] for d in data[dialogue_i]['dialog']]
                wizard_utterances = speakers.count('1_Wizard') + speakers.count('0_Wizard')
                predict_wizard = predict_wizards(wizard_utterances)

                for predict_wizard_i in predict_wizard:
                    for d in data[dialogue_i]['dialog']:
                        # print('-' * 39)
                        if 'Wizard' in d['speaker']:
                            wizard_count += 1
                            target = d['text']

                        flattened_knowledges = [k for dk in dialogue_knowledges[-n_utterances_back:] for k in dk]
                        knowledge = ['no passages used'] + chosen_topic_passage + flattened_knowledges + zero_knowledge
                        knowledge = knowledge[:max_knowledge]  # [:16]
                        random.shuffle(knowledge)

                        rp = [list(l.keys())[0] + ': ' + ' '.join(list(l.values())[0]) for l in d['retrieved_passages']]
                        dialogue_knowledges.append(rp)

                        if 'checked_sentence' in d.keys():
                            chosen = list(d['checked_sentence'].values())[0] \
                                if not len(d['checked_sentence']) == 0 else 'no passages used'
                            chosen_i = None
                            for i, s in enumerate(knowledge):
                                if chosen.replace('_', ' ') in s.replace('_', ' '):
                                    chosen_i = i

                            try:
                                assert not chosen_i is None
                            except:
                                chosen_not_found += 1
                                chosen_i = knowledge.index('no passages used')

                        if wizard_count > predict_wizard_i: break
                        speaker = 'me' if 'Wizard' in d['speaker'] else 'you'
                        context += ' {}: {}'.format(speaker, d['text'])

                    # print(target)
                    # print(context)
                    # print('Target:')
                    output = tokenize(target, tokenizer, tokenizer_choice)
                    target_length = len(output)
                    targets.append(output[:safe_max_len])

                    # print('Context:')
                    output = tokenize(context, tokenizer, tokenizer_choice)
                    context_length = len(output)
                    contexts.append(output[-safe_max_len:])

                    # print('Knowledge:')
                    k_ids = [tokenize(k, tokenizer, tokenizer_choice)[-safe_max_len:] for k in knowledge]
                    knowledge_lengths = [len(k) for k in k_ids]
                    if not len(knowledge_lengths) == 32:
                        print(chosen_topic_passage)
                        print(len(knowledge_lengths))
                    knowledges.append(k_ids)

                    # print('Knowledge id:')
                    choices.append(chosen_i)

                    max_target_length = max(target_length, max_target_length)
                    max_context_length = max(context_length, max_context_length)
                    max_knowledge_length = max(max(knowledge_lengths), max_knowledge_length)
                    max_knowledge_items = max(len(knowledge_lengths), max_knowledge_items)

            targets = pad_sequences(targets, padding='post', value=pad_idx)[:, :safe_max_len]
            contexts = pad_sequences(contexts, value=pad_idx)[:, -safe_max_len:]
            choices = np.array(choices)[..., None]

            l_0, l_1 = len(knowledges), len(knowledges[0])
            b = pad_sequences([l[-safe_max_len:] for li in knowledges for l in li], value=pad_idx)
            knowledges = np.reshape(b, (l_0, l_1, -1))

            df = pd.DataFrame(
                np.array(
                    [split_name, max_target_length, max_context_length, max_knowledge_length, max_knowledge_items,
                     len(targets), len(data), chosen_not_found])[None],
                columns=['data_split', 'max_target_length', 'max_context_length', 'max_knowledge_length',
                         'max_knowledge_items', 'n_samples', 'n_dialogues', 'chosen_not_found'])
            large_df = large_df.append(df)
            data_lists = {'targets': targets, 'contexts': contexts, 'choices': choices, 'knowledges': knowledges}

            shuffled_indices = np.random.choice(l_0, l_0, replace=False)

            with h5py.File(h5_path, 'w') as f:
                for k, v in data_lists.items():
                    f.create_dataset(k, data=v[shuffled_indices][::reduce_data_by])

    if large_df.shape[0] == 5:
        print(large_df)
        with open(os.path.join(DATADESTINATION, 'summary.txt'), 'w') as f:
            f.write(large_df.to_string(index=False))


def test_load():
    split_name = 'valid_random_split'
    h5_path = os.path.join(DATAPATH, '{}_{}.h5'.format(split_name, args.tokenizer_choice))

    tokenizer_path = os.path.join(DATAPATH, 'tokenizer-bpe.json')
    tokenizer = Tokenizer.from_file(tokenizer_path)

    f = h5py.File(h5_path, 'r')
    print(f.keys())
    batch_size = 4
    pad_idx = tokenizer.encode('[PAD]').ids[0]
    batch_indices = sorted(np.random.choice(10, batch_size, replace=False))
    reshuffled_indices = np.random.choice(batch_size, batch_size, replace=False)
    print(batch_indices)

    batch = f['targets'][batch_indices]
    batch = [eval(s) for s in batch]
    padded = pad_sequences(batch, value=pad_idx)[reshuffled_indices]

    choices = np.array([eval(s) for s in f['choices'][batch_indices]])[reshuffled_indices]

    batch = f['knowledges'][batch_indices]
    batch = [[eval(s) for s in sbatch] for sbatch in batch]
    maxlen = max([len(item) for sublist in batch for item in sublist])
    padded_knowledges = np.asarray([[[pad_idx] * (maxlen - len(s)) + s for s in sbatch] for sbatch in batch])[
        reshuffled_indices]
    print(padded_knowledges.shape)

    print()
    print(padded)
    print(choices)


class WikipediaWizardGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            encoder_maxlen=512,
            decoder_maxlen=512,
            data_split='train',
            data_path=None,
            tokenizer_choice='bpe',
            n_dialogues=-1,
    ):
        DATADESTINATION = os.path.join(data_path, tokenizer_choice)

        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            encoder_maxlen=encoder_maxlen,
            decoder_maxlen=decoder_maxlen,
            data_split=data_split,
            data_path=DATADESTINATION,
            tokenizer_choice=tokenizer_choice,
            n_dialogues=n_dialogues,
        )

        assert data_split in split_names
        if data_path is None:
            raise ValueError("Specify the data_path where you want the data to be saved!")
        download(data_path=data_path, tokenizer_choice=tokenizer_choice, n_dialogues=n_dialogues)

        self.on_epoch_end()

        tokenizer_path = os.path.join(self.data_path, 'tokenizer-{}.json'.format(tokenizer_choice))

        if tokenizer_choice == 'bpe':
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.data_path)

        pad_word = '[PAD]' if tokenizer_choice == 'bpe' else '<|endoftext|>'
        self.pad_idx = tokenize(pad_word, self.tokenizer, tokenizer_choice)[0]

        start_word = '[START]' if tokenizer_choice == 'bpe' else '<|endoftext|>'
        self.start_idx = tokenize(start_word, self.tokenizer, tokenizer_choice)[0]

        self.epochs = 50 if epochs == None else epochs

    def on_epoch_end(self):

        if hasattr(self, 'data'):
            self.data.close()
            del self.data

        h5_path = os.path.join(self.data_path, '{}_{}_{}.h5'.format(self.data_split, self.tokenizer_choice, self.n_dialogues))
        self.data = h5py.File(h5_path, 'r')
        n_samples = len(self.data['choices'])
        self.random_indices = np.random.choice(n_samples, n_samples, replace=False)

        if self.steps_per_epoch == None:
            self.steps_per_epoch = int(n_samples / self.batch_size)

    def data_generation(self, index=None):
        if index is None:
            index = np.random.randint(0, self.steps_per_epoch)
        batch_indices = self.random_indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_indices = sorted(batch_indices)
        reshuffled_indices = np.random.choice(self.batch_size, self.batch_size, replace=False)

        targets = self.data['targets'][batch_indices]
        targets = unpad_sequence(targets, padding='post', value=self.pad_idx)[reshuffled_indices]
        input_targets = self.start_idx * np.ones((targets.shape[0], targets.shape[1] + 1), dtype=np.int32)
        input_targets[:, 1:] = targets
        output_targets = self.pad_idx * np.ones((targets.shape[0], targets.shape[1] + 1), dtype=np.int32)
        output_targets[:, :-1] = targets

        contexts = self.data['contexts'][batch_indices]
        padded_contexts = unpad_sequence(contexts, padding='pre', value=self.pad_idx)[reshuffled_indices]
        choices = self.data['choices'][batch_indices][reshuffled_indices]

        knowledges = self.data['knowledges'][batch_indices]
        padded_knowledges = unpad_sequence(knowledges, padding='pre', value=self.pad_idx)[reshuffled_indices]

        return {'choices': choices, 'knowledges': padded_knowledges[..., -self.encoder_maxlen:],
                'targets': input_targets[..., :self.decoder_maxlen],
                'contexts': padded_contexts[..., -self.encoder_maxlen:],
                'output_targets': output_targets[..., :self.decoder_maxlen]}

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        batch = self.data_generation(index)
        # print(['{}: {}'.format(k, v.shape) for k, v in batch.items()])
        return [
                   batch['contexts'],
                   batch['knowledges'],
                   batch['choices'],
                   batch['targets']
               ], \
               batch['output_targets']


if __name__ == '__main__':
    CDIR = os.path.dirname(os.path.realpath(__file__))
    DATAPATH = os.path.abspath(os.path.join(CDIR, 'data', 'wizard_of_wikipedia'))
    os.makedirs(DATAPATH, exist_ok=True)

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument(
        '--tokenizer_choice', default='bpe', type=str, help='which tokenizer to use (either bpe of gpt2)',
    )
    parser.add_argument(
        '--n_dialogues', default=10, type=int,
        help='number of dialogues to tokenize, if <0 tokenizes the whole dataset',
    )
    args = parser.parse_args()

    download(data_path=DATAPATH, tokenizer_choice=args.tokenizer_choice, n_dialogues=args.n_dialogues)
