import os, shutil, json, random, h5py, argparse
from tokenizers import Tokenizer
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import numpy as np
import pandas as pd
from GenericTools.PlotTools.mpl_tools import load_plot_settings
from GenericTools.StayOrganizedTools.download_utils import download_and_unzip

pd = load_plot_settings(pd=pd)

split_names = ['valid_random_split', 'valid_topic_split', 'test_random_split', 'test_topic_split', 'train']


def download(data_path, tokenizer_choice, n_dialogues):
    n_utterances_back = 4
    max_knowledge = 32
    DATAPATH = data_path
    n_dialogues = int(n_dialogues) if n_dialogues > 0 else None
    assert tokenizer_choice in ['bpe', 'gpt2']
    tokenizer_path = os.path.join(DATAPATH, 'tokenizer-{}.json'.format(tokenizer_choice))

    if len(os.listdir(DATAPATH)) == 0:
        url = 'http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz'
        download_and_unzip([url], DATAPATH)

    if not os.path.isfile(tokenizer_path):
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace, Sequence, Digits

        url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip'
        download_and_unzip([url], DATAPATH)

        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))

        trainer = BpeTrainer(special_tokens=['[UNK]', '[WIZARD]', '[APPRENTICE]', '[PAD]', '[START]'])
        pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True)])
        tokenizer.pre_tokenizer = pre_tokenizer

        files = [os.path.join(DATAPATH, f'wikitext-103-raw/wiki.{split}.raw') for split in ['test', 'train', 'valid']]
        tokenizer.train(files, trainer)

        tokenizer.save(tokenizer_path)

        shutil.rmtree(os.path.join(DATAPATH, 'wikitext-103-raw'))
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

    large_df = pd.DataFrame(
        columns=['data_split', 'max_target_length', 'max_context_length', 'max_knowledge_length',
                 'max_knowledge_items', 'n_samples', 'n_dialogues', 'chosen_not_found'])

    for split_name in split_names:
        max_target_length, max_context_length, max_knowledge_length, max_knowledge_items = 0, 0, 0, 0
        chosen_not_found = 0

        h5_path = os.path.join(DATAPATH, '{}_{}.h5'.format(split_name, tokenizer_choice))
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
                zero_knowledge = ['[PAD][PAD]'] * 7
                dialogue_knowledges = [zero_knowledge] * n_utterances_back

                speakers = [d['speaker'] for d in data[dialogue_i]['dialog']]
                wizard_utterances = speakers.count('1_Wizard') + speakers.count('0_Wizard')
                predict_wizard_i = np.random.choice(wizard_utterances)
                for d in data[dialogue_i]['dialog']:
                    # print('-' * 39)
                    if 'Wizard' in d['speaker']:
                        wizard_count += 1
                        target = d['text']

                    flattened_knowledges = [k for dk in dialogue_knowledges[-n_utterances_back:] for k in dk]
                    print(len(flattened_knowledges))
                    knowledge = ['no passages used'] + chosen_topic_passage + flattened_knowledges + zero_knowledge
                    knowledge = knowledge[:max_knowledge]  # [:16]
                    random.shuffle(knowledge)

                    rp = [list(l.keys())[0] + ': ' + ' '.join(list(l.values())[0]) for l in d['retrieved_passages']]
                    dialogue_knowledges.append(rp)

                    print(rp)

                    if 'checked_sentence' in d.keys():
                        chosen = list(d['checked_sentence'].values())[0] \
                            if not len(d['checked_sentence']) == 0 else 'no passages used'
                        chosen_i = None
                        for i, s in enumerate(knowledge):
                            if chosen.replace('_', ' ') in s.replace('_', ' '):
                                chosen_i = i

                        print(chosen)
                        # print(knowledge)
                        try:
                            assert not chosen_i is None
                        except:
                            chosen_not_found +=1
                            chosen_i = knowledge.index('no_passages_used')

                    if wizard_count > predict_wizard_i: break
                    speaker = d['speaker'].replace('1_', '').replace('0_', '').upper()
                    context += ' [{}] {}'.format(speaker, d['text'])

                # print(target)
                # print(context)
                # print('Target:')
                output = tokenizer.encode(target)
                target_length = len(output.ids)
                targets.append(output.ids)

                # print('Context:')
                output = tokenizer.encode(context)
                context_length = len(output.ids)
                contexts.append(output.ids)

                # print('Knowledge:')
                k_ids = [tokenizer.encode(k).ids for k in knowledge]
                knowledge_lengths = [len(k) for k in k_ids]
                knowledges.append(k_ids)

                # print('Knowledge id:')
                choices.append(chosen_i)

                max_target_length = max(target_length, max_target_length)
                max_context_length = max(context_length, max_context_length)
                max_knowledge_length = max(max(knowledge_lengths), max_knowledge_length)
                max_knowledge_items = max(len(knowledge_lengths), max_knowledge_items)

            df = pd.DataFrame(
                np.array(
                    [split_name, max_target_length, max_context_length, max_knowledge_length, max_knowledge_items,
                     len(targets), len(data), chosen_not_found])[None],
                columns=['data_split', 'max_target_length', 'max_context_length', 'max_knowledge_length',
                         'max_knowledge_items', 'n_samples', 'n_dialogues', 'chosen_not_found'])
            large_df = large_df.append(df)
            data_lists = {'targets': targets, 'contexts': contexts, 'choices': choices, 'knowledges': knowledges}

            with h5py.File(h5_path, 'w') as f:
                dt = h5py.special_dtype(vlen=str)
                dsets = {k: f.create_dataset(k, data=np.array(data_lists[k], dtype=object), dtype=dt) for k in
                         data_lists.keys()}

    if large_df.shape[0] == 5:
        print(large_df)
        with open(os.path.join(DATAPATH, 'summary.txt'), 'w') as f:
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
            maxlen=512,
            data_split='train',
            data_path=None,
            tokenizer_choice='bpe',
            n_dialogues=-1,
    ):

        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            maxlen=maxlen,
            data_split=data_split,
            data_path=data_path,
            tokenizer_choice=tokenizer_choice,
        )

        assert data_split in split_names
        if data_path is None:
            raise ValueError("Specify the data_path where you want the data to be saved!")
        download(data_path=data_path, tokenizer_choice=tokenizer_choice, n_dialogues=n_dialogues)

        self.maxlen = maxlen
        self.on_epoch_end()

        tokenizer_path = os.path.join(self.data_path, 'tokenizer-{}.json'.format(tokenizer_choice))
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_idx = self.tokenizer.encode('[PAD]').ids[0]
        self.start_idx = self.tokenizer.encode('[START]').ids[0]

        self.epochs = 50 if epochs == None else epochs

    def on_epoch_end(self):

        if hasattr(self, 'data'):
            self.data.close()
            del self.data

        h5_path = os.path.join(self.data_path, '{}_{}.h5'.format(self.data_split, self.tokenizer_choice))
        self.data = h5py.File(h5_path, 'r')
        n_samples = len(self.data['choices'])
        self.random_indices = np.random.choice(n_samples, n_samples, replace=False)

        if self.steps_per_epoch == None:
            self.steps_per_epoch = int(n_samples / self.batch_size)

    def data_generation(self, index=None):
        if index is None:
            index = np.random.randint(0, self.steps_per_epoch)

        batch_indices = self.random_indices[index * self.batch_size:(index + 1) * self.batch_size]
        # self.random_indices = self.random_indices[:-self.batch_size]
        batch_indices = sorted(batch_indices)
        reshuffled_indices = np.random.choice(self.batch_size, self.batch_size, replace=False)

        targets = [eval(s) for s in self.data['targets'][batch_indices]]
        input_targets = [[self.start_idx] + s for s in targets]
        output_targets = [s + [self.pad_idx] for s in targets]
        input_targets = pad_sequences(input_targets, value=self.pad_idx)[reshuffled_indices]
        output_targets = pad_sequences(output_targets, value=self.pad_idx)[reshuffled_indices]

        contexts = [eval(s) for s in self.data['contexts'][batch_indices]]
        padded_contexts = pad_sequences(contexts, value=self.pad_idx)[reshuffled_indices]

        choices = np.array([eval(s) for s in self.data['choices'][batch_indices]])[reshuffled_indices]

        knowledges = [[[int(i) for i in s[1:-1].split(',')] for s in k] for k in self.data['knowledges'][batch_indices]]
        maxlen = max([len(item) for sublist in knowledges for item in sublist])
        padded_knowledges = [[[self.pad_idx] * (maxlen - len(s)) + s for s in k] for k in knowledges]
        padded_knowledges = np.asarray(padded_knowledges)[reshuffled_indices]

        return {'choices': choices, 'knowledges': padded_knowledges[..., -self.maxlen:],
                'targets': input_targets[..., -self.maxlen:], 'contexts': padded_contexts[..., -self.maxlen:],
                'output_targets': output_targets[..., -self.maxlen:]}

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        batch = self.data_generation(index)
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
