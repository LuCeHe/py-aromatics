'''
sources:
https://raw.githubusercontent.com/jarednielsen/deep-learning-models/nlp/models/nlp/common/preprocess.py
https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py

'''

import argparse, multiprocessing, os, time, shutil
import datasets as nlp
from transformers import GPT2Tokenizer

CDIR = os.path.dirname(os.path.realpath(__file__))


def text_to_language_modeling_tokenization(datapath, dataset, data_split, max_seq_length, preprocessing_num_workers):
    import time
    dataset_path = os.path.join(datapath, dataset)
    setpath = os.path.join(dataset_path, data_split)
    set_tokenized_path = os.path.join(dataset_path, 'tokenized_{}_{}'.format(data_split, max_seq_length))

    start_time = time.perf_counter()

    text_column_name = 'text'

    # if not already tokenized
    if not os.path.isdir(set_tokenized_path):
        print('\nLoading dataset...')

        # if not already downloaded
        if True: #not os.path.isdir(setpath):
            os.makedirs(setpath, exist_ok=True)

            print(f'Loading dataset: {dataset}')
            if dataset.startswith('wikitext'):
                dset = nlp.load_dataset(
                    'wikitext', f'{dataset}-raw-v1', split=data_split, cache_dir=setpath
                )

            elif dataset == 'wikipedia':
                dset = nlp.load_dataset('wikipedia', '20200501.en', split=data_split, cache_dir=setpath)
                dset.remove_columns_('title')  # only keep the text

            elif dataset == 'bookcorpus':
                dset = nlp.load_dataset('bookcorpus', split=data_split, cache_dir=setpath, ignore_verifications=True)

            elif dataset == 'wikibooks':

                bookcorpus = nlp.load_dataset('bookcorpus', split='train', cache_dir=setpath)
                wiki = nlp.load_dataset('wikipedia', '20200501.en', split='train', cache_dir=setpath)
                wiki.remove_columns_('title')  # only keep the text
                assert bookcorpus.features.type == wiki.features.type
                dset = nlp.concatenate_datasets([bookcorpus, wiki])

            elif dataset == 'c4':
                dset = nlp.load_dataset('c4', 'en', cache_dir=setpath)
                # assert False, 'This dataset must be preprocessed beforehand'

            elif dataset == 'squad':
                dset = nlp.load_dataset(dataset, split=data_split, cache_dir=setpath)
                dset.remove_columns_(['title', 'answers', 'id', 'question', ])
                dset.rename_column_('context', 'text')

            elif dataset == 'lambada':
                dset = nlp.load_dataset(dataset, split=data_split, cache_dir=setpath)
                dset.remove_columns_(['domain', ])

            elif dataset == 'openwebtext':
                dset = nlp.load_dataset(dataset, split=data_split, cache_dir=setpath)
                print(dset.column_names)
                # dset.remove_columns_(['title', 'answers', 'id', 'question', ])
                # dset.rename_column_('context', 'text')
            else:
                dset = nlp.load_dataset(dataset, split=data_split, cache_dir=setpath)
                print(dset.column_names)
                print(dset[0])
                # raise NotImplementedError

            dset.save_to_disk(setpath)
        else:
            dset = nlp.load_from_disk(setpath)
            print(dset[0])
            print(dset.column_names)

        print(dset.column_names)

        print('\nLoaded dataset:')
        print('                     ', dset)
        print('                     ', dset[0])
        print('                     ', dset[1])
        assert dset.column_names == [text_column_name], "Dataset should have exactly one 'text' column"

        # dset = dset.filter(
        #     lambda ex: len(ex[text_column_name]) > 0,
        #     keep_in_memory=True,
        #     load_from_cache_file=True,
        #     num_proc=preprocessing_num_workers,
        # )
        # print('\nFiltered empty lines:')
        # print('                     ', dset)
        # print('                     ', dset[0])
        # print('                     ', dset[1])

        # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        print('\nTokenizing dataset...')

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        dset = dset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=['text']
        )

        del tokenizer
        print('\nTokenized:')
        print('                     ', dset)
        print('                     ', dset[0])
        print('                     ', dset[1])

        preprocessing_batch_size = 1500  # 300000

        print('\nStructure for Language Modeling...')

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples['input_ids'])

            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length

            # Split by chunks of max_len.
            result = {
                k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        dset = dset.map(
            group_texts,
            batch_size=preprocessing_batch_size,
            writer_batch_size=preprocessing_batch_size,
            batched=True,
            num_proc=preprocessing_num_workers,
        )

        dset.save_to_disk(set_tokenized_path)
    else:
        dset = nlp.load_from_disk(set_tokenized_path)

    print('\nGrouped as LM:')
    print('                     ', dset)
    print('                     ', dset[0])
    print('                     ', dset[1])

    elapsed = time.perf_counter() - start_time
    print(f'\nTotal processing time: {elapsed:.3f} seconds')

    del dset
    time.sleep(2)
    ds = [d for d in os.listdir(dataset_path) if not 'tokenized' in d]
    print(ds)
    for d in ds:
        d_path = os.path.join(dataset_path, d)
        if os.path.isfile(d_path):
            os.remove(d_path)
        else:
            shutil.rmtree(d_path)


if __name__ == '__main__':
    dataset = 'squad'
    DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data'))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', default=dataset,
        choices=['wikitext-2', 'wikitext-103', 'wikipedia', 'bookcorpus', 'wikibooks', 'c4', 'squad', 'openwebtext',
                 'lambada'],
    )
    parser.add_argument('--datapath', default=DATAPATH)
    parser.add_argument('--data_split', choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument('--max_seq_length', type=int, default=1024)
    parser.add_argument('--preprocessing_num_workers', type=int, default=1)

    args = parser.parse_args()

    text_to_language_modeling_tokenization(args.datapath, args.dataset, args.data_split, args.max_seq_length,
                                           args.preprocessing_num_workers)
