import os, shutil, json, random, h5py, argparse

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import numpy as np
import pandas as pd

from GenericTools.PlotTools.mpl_tools import load_plot_settings

pd = load_plot_settings(pd=pd)

parser = argparse.ArgumentParser(description='main')
parser.add_argument(
    '--tokenizer_choice', default='bpe', type=str, help='which tokenizer to use (either bpe of gpt2)',
)
parser.add_argument(
    '--n_dialogues', default=10, type=int, help='number of dialogues to tokenize, if <0 tokenizes the whole dataset',
)
args = parser.parse_args()

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Sequence, Digits

from GenericTools.StayOrganizedTools.download_utils import download_and_unzip

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, 'data', 'wizard_of_wikipedia'))
# if not os.path.isdir(DATAPATH): os.mkdir(DATAPATH)
os.makedirs(DATAPATH, exist_ok=True)

# longest dialogue: 23 utterances in train

assert args.tokenizer_choice in ['bpe', 'gpt2']
# tokenizer_choice = 'bpe'  # gpt2
show_dialogue = True

split_names = ['valid_random_split', 'valid_topic_split', 'train', 'test_random_split', 'test_topic_split']
split_name = 'valid_random_split'
n_dialogues = int(args.n_dialogues) if args.n_dialogues > 0 else None

tokenizer_path = os.path.join(DATAPATH, 'tokenizer-{}.json'.format(args.tokenizer_choice))

if len(os.listdir(DATAPATH)) == 0:
    url = 'http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz'
    download_and_unzip([url], DATAPATH)

if not os.path.isfile(tokenizer_path):

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
             'max_knowledge_items', 'n_samples', 'n_dialogues'])

for split_name in split_names:
    max_target_length, max_context_length, max_knowledge_length, max_knowledge_items = 0, 0, 0, 0
    h5_path = os.path.join(DATAPATH, '{}_{}.h5'.format(split_name, args.tokenizer_choice))
    data_json = os.path.join(DATAPATH, split_name + '.json')

    if not os.path.isfile(h5_path):
        with open(data_json) as f:
            data = json.load(f)

        targets = []
        contexts = []
        knowledges = []
        choices = []
        for dialogue_i in range(len(data))[:n_dialogues]:
            # print('-' * 59, dialogue_i)
            context = data[dialogue_i]['chosen_topic']
            target = ''
            wizard_count = 0
            chosen_topic_passage = [
                data[dialogue_i]['chosen_topic'] + ': ' + ' '.join(data[dialogue_i]['chosen_topic_passage'])]
            knowledge_1_back = []
            knowledge_2_back = []

            # print('Length conversation: ', len(data[dialogue_i]['dialog']))
            speakers = [d['speaker'] for d in data[dialogue_i]['dialog']]
            # print(speakers)
            wizard_utterances = speakers.count('1_Wizard') + speakers.count('0_Wizard')
            # print('Wizard speaks {} times'.format(wizard_utterances))
            predict_wizard_i = np.random.choice(wizard_utterances)
            # print('Wizard to predict: ', predict_wizard_i)
            for d in tqdm(data[dialogue_i]['dialog']):
                # print('-' * 39)
                # print(d.keys())
                # print(d['speaker'])
                if 'Wizard' in d['speaker']:
                    wizard_count += 1
                    target = d['text']
                rp = [list(l.keys())[0] + ': ' + ' '.join(list(l.values())[0]) for l in d['retrieved_passages']]

                knowledge = ['no passages used'] + chosen_topic_passage + knowledge_1_back + knowledge_2_back
                random.shuffle(knowledge)
                knowledge_2_back = knowledge_1_back
                knowledge_1_back = rp

                if 'checked_sentence' in d.keys():
                    chosen = list(d['checked_sentence'].values())[0] \
                        if not len(d['checked_sentence']) == 0 else 'no passages used'
                    chosen_i = None
                    for i, s in enumerate(knowledge):
                        if chosen.replace('_', ' ') in s.replace('_', ' '):
                            chosen_i = i

                    assert not chosen_i is None

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
                 len(targets), len(data)])[None],
            columns=['data_split', 'max_target_length', 'max_context_length', 'max_knowledge_length',
                     'max_knowledge_items', 'n_samples', 'n_dialogues'])
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



f = h5py.File(h5_path, 'r')
print(f.keys())
batch_size = 4
pad_idx = tokenizer.encode('[PAD]').ids
batch_indices = sorted(np.random.choice(10, batch_size, replace=False))
reshuffled_indices = np.random.choice(batch_size, batch_size, replace=False)
print(batch_indices)

batch = f['targets'][batch_indices]
batch = [eval(s) for s in batch]
padded = pad_sequences(batch, value=pad_idx)[reshuffled_indices]

choices = np.array([eval(s) for s in f['choices'][batch_indices]])[reshuffled_indices]

print()
print(padded)
print(choices)


batch = f['knowledges'][batch_indices]
batch = [eval(s) for s in batch]
maxlen = max([len(item) for sublist in batch for item in sublist])
print(maxlen)
# padded = pad_sequences(batch, value=pad_idx)[reshuffled_indices]
