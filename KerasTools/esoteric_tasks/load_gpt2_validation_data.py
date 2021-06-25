import os, argparse

from GenericTools.KerasTools.lm_preprocessing import text_to_language_modeling_tokenization

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', 'data'))
datasets = ["wikitext-2", "wikitext-103", 'squad', 'lambada', 'openwebtext', 'bookcorpus']

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", default='bookcorpus',
    choices=datasets,
)

parser.add_argument("--datapath", default=DATAPATH)
parser.add_argument("--data_split", choices=["train", "validation", "test"], default='validation')
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--preprocessing_num_workers", type=int, default=4)

args = parser.parse_args()

text_to_language_modeling_tokenization(DATAPATH, args.dataset, args.data_split, args.max_seq_length,
                                       args.preprocessing_num_workers)