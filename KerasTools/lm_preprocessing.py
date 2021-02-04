"""
sources:
https://raw.githubusercontent.com/jarednielsen/deep-learning-models/nlp/models/nlp/common/preprocess.py
https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py

"""

import argparse, multiprocessing, os, time
import datasets as nlp
from transformers import GPT2Tokenizer

CDIR = os.path.dirname(os.path.realpath(__file__))
dataset = 'squad'
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data', dataset))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", type=int, default=64)
parser.add_argument(
    "--dataset",
    choices=["wikitext-2", "wikitext-103", "wikipedia", "bookcorpus", "wikibooks", "c4", 'squad'],
    default=dataset
)

parser.add_argument("--data_split", choices=["train", "validation", "test"], default='train')
parser.add_argument("--cache_dir", default=DATAPATH)
parser.add_argument("--shards", type=int, default=1)
parser.add_argument("--processes", type=int, default=1)  # 64 processes is a sweet spot on p3dn
parser.add_argument("--skip_load_from_cache_file", action="store_true")
parser.add_argument("--preprocessing_num_workers", type=int, default=1)

args = parser.parse_args()

SETPATH = os.path.join(DATAPATH, args.data_split)

FILTER_CACHE = "filterlines.arrow"
NEWLINES_CACHE = "replacenewlines.arrow"
SENTENCES_CACHE = "sentences.arrow"
PRETOKENIZED_SENTENCES_CACHE = "pretokenized_sentences.arrow"
EXAMPLES_CACHE = f"examples_{args.max_seq_length}seq.arrow"
EXAMPLE_IDS_CACHE = f"example_ids_{args.max_seq_length}seq.arrow"

load_from_cache_file = not args.skip_load_from_cache_file

assert (
        args.dataset in args.cache_dir
), "Dataset name should be part of the directory name, don't mix datasets!"

start_time = time.perf_counter()

text_column_name = 'text'
if not os.path.isdir(SETPATH):
    os.makedirs(SETPATH, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    if args.dataset.startswith("wikitext"):
        dset = nlp.load_dataset(
            "wikitext", f"{args.dataset}-raw-v1", split=args.data_split, cache_dir=SETPATH
        )

    elif args.dataset == "wikipedia":
        dset = nlp.load_dataset("wikipedia", "20200501.en", split=args.data_split, cache_dir=SETPATH)
        dset.remove_columns_("title")  # only keep the text

    elif args.dataset == "bookcorpus":

        dset = nlp.load_dataset("bookcorpus", split=args.data_split, cache_dir=SETPATH)
        dset.remove_columns_("title")  # only keep the text

    elif args.dataset == "wikibooks":

        bookcorpus = nlp.load_dataset("bookcorpus", split="train", cache_dir=SETPATH)
        wiki = nlp.load_dataset("wikipedia", "20200501.en", split="train", cache_dir=SETPATH)
        wiki.remove_columns_("title")  # only keep the text
        assert bookcorpus.features.type == wiki.features.type
        dset = nlp.concatenate_datasets([bookcorpus, wiki])

    elif args.dataset == "c4":
        dset = nlp.load_dataset("c4", "en", cache_dir=SETPATH)
        # assert False, "This dataset must be preprocessed beforehand"

    elif args.dataset == "squad":
        dset = nlp.load_dataset(args.dataset, split=args.data_split, cache_dir=SETPATH)
        dset.remove_columns_(["title", 'answers', 'id', 'question', ])
        dset.rename_column_('context', 'text')

    else:
        raise NotImplementedError

    dset.save_to_disk(SETPATH)
else:
    dset = nlp.load_from_disk(SETPATH)

print(dset.column_names)

print("Loaded dataset:", dset, dset[0])
assert dset.column_names == [text_column_name], "Dataset should have exactly one 'text' column"

print("Filtering empty lines")
dset = dset.filter(
    lambda ex: len(ex[text_column_name]) > 0,
    cache_file_name=os.path.join(args.cache_dir, FILTER_CACHE),
    load_from_cache_file=load_from_cache_file,
)
print("Filtered empty lines:", dset, dset[0])
print("                     ", dset[1])

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def tokenize_function(examples):
    return tokenizer(examples[text_column_name], return_special_tokens_mask=True)


dset = dset.map(
    tokenize_function,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    load_from_cache_file=True,
)

print("Filtered empty lines:", dset, dset[0])
print("                     ", dset[1])
max_seq_length = args.max_seq_length


preprocessing_batch_size = 1500 #300000
def group_texts(examples):
    # Concatenate all texts.
    # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # print(examples['input_ids'])
    concatenated_examples = {'input_ids': sum(examples['input_ids'], [])}
    total_length = len(concatenated_examples['input_ids'])
    print('\nhere')
    print(total_length)

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    print(total_length)

    # Split by chunks of max_len.
    result = {
        k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)][:preprocessing_batch_size]
        for k, t in concatenated_examples.items()
    }
    return result

# batch_size
# writer_batch_size
dset = dset.map(
    group_texts,
    batch_size=preprocessing_batch_size,
    writer_batch_size=preprocessing_batch_size,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    # load_from_cache_file=not data_args.overwrite_cache,
)

print("Filtered empty lines:", dset, dset[0])
print("                     ", dset[1])

elapsed = time.perf_counter() - start_time
print(f"Total processing time: {elapsed:.3f} seconds")
