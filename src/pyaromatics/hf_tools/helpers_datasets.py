import os, argparse, sys, socket, random, itertools, gc, json, re, hashlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# from bridge_official.paths import DATADIR, cachedir, WORKDIR, CHECKPOINTS


# HFDIR = os.path.join(DATADIR, 'hf_cache')
# os.environ['HF_HOME'] = HFDIR
# os.environ['HF_DATASETS_CACHE'] = HFDIR
# os.system(f"export HF_HOME={HFDIR}")

from glob import glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import AutoConfig, Qwen3Config

import datasets
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, IterableDataset, load_from_disk, \
    Features, Value
from lm_eval import simple_evaluate
from lm_eval.api.registry import get_model as get_lm_eval_model
from lm_eval.models.huggingface import HFLM

from pyaromatics.hf_tools.utils import get_hf_key
from pyaromatics.hf_tools.utils import get_pretrained_model
from pyaromatics.hf_tools.utils import get_tokenizer as hf_get_tokenizer
from pyaromatics.stay_organized.utils import str2val
from pyaromatics.hf_tools.dataset_tools.rule110.rule110 import generate_rule110_dataset
from pyaromatics.hf_tools.dataset_tools.mqar.mqar import build_mqar_dataset_dict
from pyaromatics.stay_organized.utils import NumpyEncoder

winogrande_subsets = [
    'winogrande_xs', 'winogrande_s', 'winogrande_m', 'winogrande_l', 'winogrande_xl', 'winogrande_debiased'
]
arc_subsets = ['ARC-Easy', 'ARC-Challenge']

MAX_CHARS = 4_000  # safe for 1 GPU
MAX_TOTAL_CHARS = 2_000_000  # safe for 8 GPUs
SAVE_EVERY = 3_000


def get_dataset(
        dataset_name, cachedir=None, seed=42, lengths=None, notes='', retries=3, n_samples=-1, no_print=False,
        model_id=None
):
    n_samples = str2val(notes, 'nsamples', default=n_samples, output_type=int)
    # for i in range(retries):
    #     try:
    #         return get_dataset_unsafe(dataset_name, seed=seed, lengths=lengths, notes=notes, n_samples=n_samples,
    #                                   no_print=no_print, model_id=model_id)
    #     except Exception as e:
    #         print(f"Error in get_dataset_unsafe: {e}")
    #         print(f"Retrying {i + 1}/{retries}...")
    return get_dataset_unsafe(dataset_name, seed=seed, lengths=lengths, notes=notes, n_samples=n_samples,
                              no_print=no_print, model_id=model_id, cachedir=cachedir)


def get_dataset_unsafe(
        dataset_name, seed=42, lengths=None, notes='', n_samples=-1, no_print=False, model_id=None,
        cachedir=None
):
    # get_hf_key(WORKDIR)

    if 'clrs_' in dataset_name:
        dataset = get_dataset_clrs(dataset_name, seed=seed, lengths=lengths, notes=notes)

    elif dataset_name == 'ptb':
        dataset = get_dataset_ptb(notes=notes, extra_evaluation_datasets=False, cachedir=cachedir)

    elif dataset_name == 'wiki103':
        dataset = get_dataset_wiki103(cachedir=cachedir)

    elif dataset_name == 'llmmix1':
        dataset = get_lmmix1_dataset(notes=notes, cachedir=cachedir)

    elif dataset_name == 'dolma':
        dataset = get_dataset_dolma_and_tests(seed=seed, notes=notes, cachedir=cachedir)

    elif dataset_name == 'dolmas':
        dataset = get_dataset_dolma_and_tests(seed=seed, notes=notes, version='v1_6-sample', cachedir=cachedir)

    elif dataset_name == 'dolmal':
        dataset = get_dataset_dolma_and_tests(seed=seed, notes=notes, version='v1_5-sample', cachedir=cachedir)

    elif dataset_name == 'dolmax':
        dataset = get_dataset_dolma_and_tests(seed=seed, notes=notes, version='v1_7', cachedir=cachedir)

    elif dataset_name == 'fineweb':
        dataset = get_fineweb(notes=notes, cachedir=cachedir)

    elif dataset_name == 'pile':
        dataset = load_dataset("EleutherAI/pile")

    elif dataset_name == 'cnndn':
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
        dataset = dataset.rename_column("highlights", "text")

    elif dataset_name == 'xlsum':
        dataset = get_xlsum()

    elif dataset_name == 'mlsum':
        dataset = get_mlsum()

    elif dataset_name == 'rule110':
        dataset = get_rule110(model_id=model_id, notes=notes, seed=seed)

    elif re.match(r'^mqar\d+$', dataset_name):
        dataset = get_mqar_dataset(dataset_name, seed=seed, notes=notes, cachedir=cachedir)

    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")

    # datasets of interest:
    # https://huggingface.co/datasets/allenai/dolma
    # https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
    # https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T
    # https://huggingface.co/datasets/bigcode/starcoderdata

    # shuffle with seed
    for split in dataset.keys():
        dataset[split] = dataset[split].shuffle(seed=seed)

    if 'randreduce' in notes and n_samples < 0:
        # reduce by 21 times
        dataset['train'] = dataset['train'].select(
            range(0, len(dataset['train']), 21)
        )

    if n_samples > 0:
        for k, v in dataset.items():
            n = min(n_samples, len(v))
            dataset[k] = v.select(range(n))

    if 'onlytesting' in notes:
        max_samples = 4 if n_samples == -1 else min(4 * 5, n_samples)
        for k, v in dataset.items():
            dataset[k] = v.select(range(max_samples))

    if not no_print:
        for i in range(min(10, len(dataset['train']))):
            print('sample', i)
            sample = dataset['train'][i]
            for k, v in sample.items():

                if isinstance(v, list):
                    v = 'list - ' + str(v)
                text = v if len(v) < 100 else v[:100] + '...'
                text = text.replace('\n', ' ').replace('  ', ' ')
                print(f'    {k}: {text}')

    return dataset


def _mqar_cache_dir(cachedir):
    if cachedir is not None:
        return cachedir
    env = os.environ.get("HF_DATASETS_CACHE")
    if env:
        return os.path.join(env, "pyaromatics_mqar")
    return os.path.join(os.path.expanduser("~"), ".cache", "pyaromatics", "hf_datasets")


def get_mqar_dataset(dataset_name, seed=42, notes='', cachedir=None):
    """
    Synthetic MQAR (Zoology-style) data. Names ``mqar64``, ``mqar128``, … set
    ``input_seq_len`` to the trailing integer. Optional ``notes`` flags (split by
    ``_``): ``trainsamples:N``, ``evalsamples:N``, ``testsamples:N`` (default test
    3000), ``vocabsize:N``, ``numkvpairs:N``, ``numpasses:N``, ``powera:F``,
    ``trainseed:N``, ``evalseed:N``, ``testseed:N``.

    The first time a configuration is requested, the ``DatasetDict`` is written under
    ``cachedir`` (or ``HF_DATASETS_CACHE/pyaromatics_mqar`` / ``~/.cache/pyaromatics/hf_datasets``)
    and reloaded from disk on later calls.
    """
    m = re.match(r'^mqar(\d+)$', dataset_name)
    if not m:
        raise ValueError(f"Expected mqar<seq_len>, e.g. mqar64; got {dataset_name!r}")
    input_seq_len = int(m.group(1))
    train_samples = str2val(notes, 'trainsamples', default=100_000, output_type=int)
    eval_samples = str2val(notes, 'evalsamples', default=3_000, output_type=int)
    test_samples = str2val(notes, 'testsamples', default=3_000, output_type=int)
    vocab_size = str2val(notes, 'vocabsize', default=8_192, output_type=int)
    num_kv_pairs = str2val(notes, 'numkvpairs', default=8, output_type=int)
    num_passes = str2val(notes, 'numpasses', default=1, output_type=int)
    power_a = str2val(notes, 'powera', default=0.01, output_type=float)
    train_seed = str2val(notes, 'trainseed', default=seed, output_type=int)
    eval_seed = str2val(notes, 'evalseed', default=9_001, output_type=int)
    test_seed = str2val(notes, 'testseed', default=42_001, output_type=int)

    cache_key = {
        "dataset_name": dataset_name,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "test_samples": test_samples,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "test_seed": test_seed,
        "vocab_size": vocab_size,
        "input_seq_len": input_seq_len,
        "power_a": power_a,
        "num_kv_pairs": num_kv_pairs,
        "num_passes": num_passes,
        "random_non_queries": True,
    }
    digest = hashlib.sha256(
        json.dumps(cache_key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    root = _mqar_cache_dir(cachedir)
    data_path = os.path.join(root, "mqar", f"{dataset_name}_{digest}")

    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        dsd = build_mqar_dataset_dict(
            train_samples=train_samples,
            eval_samples=eval_samples,
            test_samples=test_samples,
            train_seed=train_seed,
            eval_seed=eval_seed,
            test_seed=test_seed,
            vocab_size=vocab_size,
            input_seq_len=input_seq_len,
            power_a=power_a,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
            random_non_queries=True,
        )
        dsd.save_to_disk(data_path)
    return DatasetDict.load_from_disk(data_path)


def get_rule110(model_id=None, notes='', seed=42, cachedir=None):
    from thepebbletrail_official.neuron_utils.helpers_models import model_ids
    if model_id in model_ids.keys():
        model_id = model_ids[model_id]

    tokenizer = hf_get_tokenizer(model_id, save_dir=DATADIR, notes=notes)
    vocab_size = tokenizer.vocab_size
    try:
        get_hf_key(WORKDIR)
        config = Qwen3Config()
        vocab_size = min(config.vocab_size, vocab_size)
    except Exception as e:
        print(f"Could not get HF key or config: {e}")
    vocab = list(range(vocab_size))

    max_length = tokenizer.model_max_length
    max_length = 1024

    ds = {}
    for k in ['train', 'validation', 'test']:
        dataset = generate_rule110_dataset(
            vocab=vocab,
            num_samples=1000,
            max_length=max_length,
            seed=seed
        )
        ds[k] = dataset
    dataset = DatasetDict(ds)
    return dataset


def get_xlsum(cachedir=None):
    data_path = os.path.join(cachedir, 'xlsum')
    if not os.path.exists(data_path):
        languages = [
            "amharic", "arabic", "azerbaijani", "bengali", "burmese", "chinese_simplified", "chinese_traditional",
            "english", "french", "gujarati", "hausa", "hindi", "igbo", "indonesian", "japanese", "kirundi", "korean",
            "kyrgyz", "marathi", "nepali", "oromo", "pashto", "persian", "pidgin", "portuguese", "punjabi", "russian",
            "scottish_gaelic", "serbian_cyrillic", "serbian_latin", "sinhala", "somali", "spanish", "swahili", "tamil",
            "telugu", "thai", "tigrinya", "turkish", "ukrainian", "urdu", "uzbek", "vietnamese", "welsh", "yoruba"
        ]

        splits = ["train", "validation", "test"]
        multilingual_dict = {}

        for split in splits:
            print('Loading split', split)
            split_datasets = [
                load_dataset("GEM/xlsum", name=lang, split=split, trust_remote_code=True)
                for lang in languages
            ]
            multilingual_dict[split] = concatenate_datasets(split_datasets)

        dataset = DatasetDict(multilingual_dict)
        dataset.save_to_disk(data_path)

    dataset = datasets.load_from_disk(data_path)

    dataset = dataset.rename_column("text", "article")
    dataset = dataset.rename_column("target", "text")

    return dataset


def get_mlsum(cachedir=None):
    data_path = os.path.join(cachedir, 'mlsum')
    if not os.path.exists(data_path):

        # load all languages
        lang_ds = []
        for language in ['de', 'es', 'fr', 'ru', 'tu']:
            print('Loading language', language)
            dataset = load_dataset("reciTAL/mlsum", language, trust_remote_code=True)
            print(dataset)
            print(dataset['train'][0])
            lang_ds.append(dataset)

        multilingual_dict = {}
        for split in lang_ds[0].keys():
            print('Merging split', split)
            multilingual_dict[split] = concatenate_datasets([ds[split] for ds in lang_ds])

        dataset = DatasetDict(multilingual_dict)
        dataset.save_to_disk(data_path)

    dataset = datasets.load_from_disk(data_path)

    dataset = dataset.rename_column("text", "article")
    dataset = dataset.rename_column("summary", "text")

    return dataset


def get_dataset_lambada(notes=None, cachedir=None):
    data_path = os.path.join(cachedir, 'lambada')
    if not os.path.exists(data_path):
        dataset = load_dataset("cimec/lambada", trust_remote_code=True)
        dataset.save_to_disk(data_path)

    dataset = datasets.load_from_disk(data_path)

    return dataset


def get_dataset_piqa(notes='', cachedir=None):
    data_path = os.path.join(cachedir, 'piqa')
    if not os.path.exists(data_path):
        dataset = load_dataset("ybisk/piqa", trust_remote_code=True)

        # turn the columns 'goal', 'sol1', 'sol2', and label '1'
        # into 'text': Goal: ... Solution 1: ... Solution 2: ... Answer: ...
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                lambda x: {
                    'text':
                        'Goal: ' + x['goal']
                        + '... Solution 1: ' + x['sol1']
                        + '. Solution 2: ' + x['sol2']
                        + '. Answer: ' + str(x['label'] + 1)
                        + '. Done.'
                }

            )

        # remove excess columns
        dataset = dataset.remove_columns(['goal', 'sol1', 'sol2', 'label'])
        dataset.save_to_disk(data_path)

    dataset = datasets.load_from_disk(data_path)
    return dataset


def get_dataset_hellaswag(notes='', cachedir=None):
    data_path = os.path.join(cachedir, 'hellaswag')
    if not os.path.exists(data_path):
        dataset = load_dataset("Rowan/hellaswag", trust_remote_code=True)

        # turn the columns 'goal', 'sol1', 'sol2', and label '1'
        # into 'text': Goal: ... Solution 1: ... Solution 2: ... Answer: ...
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                lambda x: {
                    'text':
                        'Activity: ' + x['activity_label']
                        + '. Context: ' + x['ctx']
                        + '... Ending 1: ' + x['endings'][0]
                        + ' Ending 2: ' + x['endings'][1]
                        + ' Ending 3: ' + x['endings'][2]
                        + ' Answer: ' + str(x['label'])
                        + '. Done.'
                }
            )
        dataset = dataset.remove_columns([
            'activity_label', 'ctx', 'endings', 'label', 'ctx_a', 'ctx_b', 'source_id', 'ind', 'split', 'split_type'
        ])
        dataset.save_to_disk(data_path)

    dataset = datasets.load_from_disk(data_path)
    return dataset


def get_dataset_arcs(notes='', subset: str = 'ARC-Challenge', cachedir=None):
    assert subset in arc_subsets
    data_path = os.path.join(cachedir, subset)
    if not os.path.exists(data_path):
        dataset = load_dataset("allenai/ai2_arc", subset, trust_remote_code=True)

        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                lambda x: {
                    'text':
                        'Question: ' + (x['question'] if x['question'].endswith('?') else x['question'] + '?')
                        + '.'.join([f' Choice {l}: {t}' for t, l in zip(x['choices']['text'], x['choices']['label'])])
                        + ' Answer: ' + x['answerKey']
                        + '. Done.'
                }
            )
        dataset = dataset.remove_columns([
            'id', 'question', 'choices', 'answerKey'
        ])
        dataset.save_to_disk(data_path)

    dataset = datasets.load_from_disk(data_path)
    return dataset


def get_dataset_winogrande(notes='', subset: str = 'winogrande_xs', cachedir=None):
    assert subset in winogrande_subsets
    data_path = os.path.join(cachedir, subset)
    if not os.path.exists(data_path):
        dataset = load_dataset("allenai/winogrande", subset, trust_remote_code=True)

        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                lambda x: {
                    'text':
                        'Question: ' + x['sentence']
                        + ' Choice 1: ' + x['option1']
                        + '. Choice 2: ' + x['option2']
                        + '. Answer: ' + x['answer']
                        + '. Done.'
                }
            )
        dataset = dataset.remove_columns([
            'sentence', 'option1', 'option2', 'answer'
        ])
        dataset.save_to_disk(data_path)

    dataset = datasets.load_from_disk(data_path)
    return dataset


def get_dataset_openbookqa(notes='', cachedir=None):
    data_path = os.path.join(cachedir, 'openbookqa')
    if not os.path.exists(data_path):
        dataset = load_dataset("allenai/openbookqa", trust_remote_code=True)

        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                lambda x: {
                    'text':
                        'Question: ' + x['question_stem'] + '...'
                        + '.'.join([f' Choice {l}: {t}' for t, l in zip(x['choices']['text'], x['choices']['label'])])
                        + '. Answer: ' + x['answerKey']
                        + '. Done.'
                }
            )
        dataset = dataset.remove_columns([
            'question_stem', 'choices', 'answerKey', 'id'
        ])
        dataset.save_to_disk(data_path)

    dataset = datasets.load_from_disk(data_path)
    return dataset


def get_dataset_ptb(notes='', extra_evaluation_datasets=False, cachedir=None):
    data_path = os.path.join(cachedir, 'ptb')
    if not os.path.exists(data_path):
        try:
            dataset = load_dataset("ptb-text-only/ptb_text_only", trust_remote_code=True)
        except Exception as e:
            print(f"Error loading ptb-text-only/ptb_text_only: {e}")
            print("Falling back to penn_treebank")
            dataset = load_dataset("FALcon6/ptb_text_only", trust_remote_code=True)

        # change 'sentence' column to 'text'
        dataset = dataset.rename_column('sentence', 'text')
        dataset.save_to_disk(data_path)

    dataset = datasets.load_from_disk(data_path)

    if extra_evaluation_datasets:
        lambada_ds = get_dataset_lambada()

        all_ds = {
            **{k: v for k, v in dataset.items()},
            **{'lambada_' + k: v for k, v in lambada_ds.items() if not 'train' in k}
        }

        dataset = DatasetDict(all_ds)

    return dataset


def get_dataset_wiki103(cachedir=None):
    data_path = os.path.join(cachedir, 'wiki103')
    if not os.path.exists(data_path):
        # Salesforce/wikitext
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

        # remove empty text
        for split in dataset.keys():
            dataset[split] = dataset[split].filter(lambda x: len(x['text'].strip()) > 0)

        dataset.save_to_disk(data_path)

    dataset = datasets.load_from_disk(data_path)
    return dataset


def get_dataset_clrs(
        dataset_name, train_samples=1000, seed=42, val_samples=1000, test_samples=1000,
        lengths=None, notes='', cachedir=None
):
    from thepebbletrail_official.dataset_utils.clrs._src.clrs_text.huggingface_generators import clrs_generator
    from thepebbletrail_official.dataset_utils.clrs._src.specs import CLRS_30_ALGS
    CLRSDIR = os.path.join(cachedir, 'clrs_datasets')
    if 'clrs_' not in dataset_name:
        raise ValueError(f"Dataset {dataset_name} not recognized.")

    dataset_name = dataset_name.replace('clrs_', '')
    dataset_path = os.path.join(CLRSDIR, dataset_name)

    lengths = lengths or [4]
    # lengths = lengths or ['leq4', 16, 64, 256, 1024, 4096, 16384]
    for length in lengths:
        length_path = os.path.join(dataset_path, str(length))
        if not os.path.exists(length_path):
            print(f'       Creating length {length}...')
            # lengths_ = [length] if not length == 'leq4' else [4, 3, 2]
            lengths_ = [length]
            if dataset_name == 'all':
                print('            all')
                algos_and_lengths = {algo: lengths_ for algo in CLRS_30_ALGS}
            elif dataset_name in CLRS_30_ALGS:
                print(f'            {dataset_name}')
                algos_and_lengths = {dataset_name: lengths_}
            else:
                raise ValueError(f"Dataset {dataset_name} not recognized.")
            num_samples = test_samples if not length == 4 else train_samples + val_samples + test_samples
            ds = Dataset.from_generator(
                clrs_generator, gen_kwargs={
                    "algos_and_lengths": algos_and_lengths,
                    "num_samples": num_samples,
                    "seed": seed,
                }
            )
            # remove all columns that are not 'text'
            ds = ds.remove_columns(["question", "answer", "algo_name", "length", "use_hints"])
            ds.save_to_disk(length_path)

    train_length = os.path.join(dataset_path, '4')
    ds = datasets.load_from_disk(train_length)

    data_names = [dataset_name]
    if dataset_name == 'all':
        data_names = ['all', 'insertion_sort']

    longer_ds = {}
    for dn in data_names:
        longers = os.listdir(os.path.join(CLRSDIR, dn))

        for longer in longers:
            if longer == '4':
                continue
            lds = datasets.load_from_disk(os.path.join(CLRSDIR, dn, longer))
            # select at max test samples
            lds = lds.select(range(test_samples))
            longer_ds[f'test_{dn}_{longer}'] = lds

    # take 1000 samples for validation and 1000 for test
    train_testvalid_split = ds.train_test_split(test_size=test_samples + val_samples, seed=seed)
    # Now split the remaining test+validation set into test (50%) and validation (50%)
    test_valid_split = train_testvalid_split['test'].train_test_split(test_size=val_samples, seed=seed)

    all_ds = {
        **longer_ds,
        'test': test_valid_split['test'],
        'validation': test_valid_split['train'],
        'train': train_testvalid_split['train'],
    }
    if 'onlytesting' in notes:
        for k, v in all_ds.items():
            all_ds[k] = v.select(range(4))

    ds = DatasetDict(all_ds)

    # standardize
    for split in ds.keys():
        # replace \n by ' ', and '  ' by ' '
        ds[split] = ds[split].map(lambda x: {'text': x['text'].replace('\n', ' ').replace('  ', ' ').strip()})
        ds[split] = ds[split].map(lambda x: {'text': 'question - ' + x['text'] + ' done.'})

        # replace last of many ':' in x['text'] with ': Answer:'
        ds[split] = ds[split].map(
            lambda x: {'text': x['text'].rsplit(':', 1)[0] + ' - answer:' + x['text'].rsplit(':', 1)[1]}
        )

    return ds


def get_dataset_dolma(notes='', version='v1_6-sample', cachedir=None):
    data_path = os.path.join(cachedir, 'dolma_' + version)
    print(data_path)

    if not os.path.exists(data_path):
        # dataset = load_dataset("allenai/dolma", 'v1_6-sample', trust_remote_code=True, streaming=True)
        # count cpus and multiply by the number of cores per cpu
        max_proc = os.cpu_count() * 4 - 1
        print('max_proc', max_proc)
        dataset = load_dataset(
            "allenai/dolma", version, trust_remote_code=True,
            # num_proc=max_proc
            # download_mode='force_redownload',
            # token=os.environ['HF_AUTH_TOKEN']
            cache_dir=data_path,  # Cache files here
        )

        dataset.save_to_disk(data_path, num_proc=1)

    dataset = datasets.load_from_disk(data_path)
    return dataset


def get_fineweb(notes='', method='normal', cachedir=None):
    data_path = os.path.join(cachedir, 'fineweb')
    print(data_path)
    if not os.path.exists(data_path):
        # count cpus and multiply by the number of cores per cpu
        max_proc = os.cpu_count() * 4 - 1
        print('max_proc', max_proc)
        dataset = load_dataset(
            "HuggingFaceFW/fineweb", trust_remote_code=True,
            num_proc=max_proc
        )

        dataset.save_to_disk(data_path)

    dataset = datasets.load_from_disk(data_path)
    return dataset


def get_dataset_dolma_and_tests(seed=0, notes='', version='v1_6-sample', cachedir=None):
    dataset = get_dataset_dolma(notes=notes, version=version, cachedir=cachedir)
    print(dataset)

    # test_names = [
    #     'lambada', 'piqa', 'hellaswag', 'openbookqa',
    #     'arcs', 'winogrande'
    # ]
    test_names = ['lambada', 'ptb']
    all_ds = {}
    for tn in test_names:
        if tn == 'arcs':
            for subset in arc_subsets:
                test_ds = globals()[f'get_dataset_{tn}'](subset=subset, notes=notes)
                for k in test_ds.keys():
                    all_ds[f'{subset.lower()}_{k}'] = test_ds[k]
        elif tn == 'winogrande':
            for subset in winogrande_subsets:
                test_ds = globals()[f'get_dataset_{tn}'](subset=subset, notes=notes)
                for k in test_ds.keys():
                    all_ds[f'{subset}_{k}'] = test_ds[k]
        else:
            test_ds = globals()[f'get_dataset_{tn}'](notes=notes)
            for k in test_ds.keys():
                all_ds[f'{tn}_{k}'] = test_ds[k]

    # create validation and test splits

    # take 1000 samples for validation and 1000 for test
    train_testvalid_split = dataset['train'].train_test_split(test_size=2000, seed=seed)
    # Now split the remaining test+validation set into test (50%) and validation (50%)
    test_valid_split = train_testvalid_split['test'].train_test_split(test_size=1000, seed=seed)

    all_ds = {
        'test': test_valid_split['test'],
        'validation': test_valid_split['train'],
        'train': train_testvalid_split['train'],
        **{k: v for k, v in all_ds.items() if not 'train' in k}
    }
    dataset = DatasetDict(all_ds)

    return dataset


def get_metrics(tokenizer=None, bos=None, eos=None, dataset_name=None):
    accs = []
    batch_sizes = []

    if dataset_name in ['ptb', 'dolma']:
        return None

    assert eos is not None and bos is not None and tokenizer is not None

    # target_vector
    answer_tokens = str(tokenizer.encode(bos)[:-1]).replace('[', '').replace(']', '')
    done_tokens = str(tokenizer.encode(eos)[:-1]).replace('[', '').replace(']', '')
    print('answer_tokens', answer_tokens)
    print('done_tokens', done_tokens)

    def compute_metrics(pred, compute_result=False):
        labels = pred.label_ids.tolist()

        # For the CLRS dataset, the answer is certain, so we can just take the argmax
        preds = pred.predictions.argmax(-1).tolist()

        # select part of the label_ids that is the answer
        ais = [str(l).split(answer_tokens)[0].count(',') + answer_tokens.count(',') + 1 for l in labels]
        dis = [str(l).split(done_tokens)[0].count(',') for l in labels]

        # select part of the predictions that is the answer
        labels = [l[ai:di] for ai, di, l in zip(ais, dis, labels)]
        preds = [p[ai:di] for ai, di, p in zip(ais, dis, preds)]

        # concatenate list of lists into a single list
        labels = [item for sublist in labels for item in sublist]
        preds = [item for sublist in preds for item in sublist]

        if len(labels) == 0:
            matches = []
            acc = 0
        else:
            matches = [1 if l == p else 0 for l, p in zip(labels, preds)]
            acc = sum(matches) / len(matches)

        accs.append(acc)
        batch_sizes.append(len(matches))

        if compute_result:
            if sum(batch_sizes) == 0:
                final_accuracy = -1
            else:
                final_accuracy = sum([a * b for a, b in zip(accs, batch_sizes)]) / sum(batch_sizes)
            accs.clear()
            batch_sizes.clear()
            return {"accuracy": final_accuracy}

        return {"accuracy": acc}

    return compute_metrics


def test_dataset():
    dataset = get_dataset(dataset_name='clrs_all', seed=0)

    # get tokenizer above

    from thepebbletrail_official.paths import DATADIR
    os.environ['HF_HOME'] = os.path.join(DATADIR, 'hf_cache')
    os.environ['HF_DATASETS_CACHE'] = os.path.join(DATADIR, 'hf_cache')
    os.system(f"export HF_HOME={os.path.join(DATADIR, 'hf_cache')}")
    from transformers import AutoTokenizer

    tokenizer_id = "google/byt5-small"

    tokenizer_path = os.path.join(DATADIR, tokenizer_id.replace('/', '-') + '-tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, truncation_side='left')

    answer_tokens = str(tokenizer.encode('answer: ')[:-1]).replace('[', '').replace(']', '')
    print(answer_tokens)
    done_tokens = str(tokenizer.encode(' done.')[:-1]).replace('[', '').replace(']', '')
    print(done_tokens)

    for i in range(10):
        print('-' * 50)
        sentence = dataset['train'][i]['text']
        print(sentence)
        tokenized = tokenizer(
            sentence, return_tensors='pt', padding='max_length', max_length=2048, truncation=True
        )

        tokenized_list = tokenized['input_ids'][0].tolist()
        str_list = str(tokenized_list)
        # answer_idx = str_list.find(answer_tokens)
        # done_idx = str_list.find(done_tokens)
        # answer = str_list.split(answer_tokens)[1].split(done_tokens)[0]
        # answer_2 = str_list[answer_idx + len(answer_tokens):done_idx]

        list_ai = str_list.split(answer_tokens)[0].count(',') + answer_tokens.count(',') + 1
        list_di = str_list.split(done_tokens)[0].count(',')

        # strip ',' and ' '
        answer = eval(answer.strip().strip(','))
        answer_2 = eval(answer_2.strip().strip(','))

        print(tokenized)
        untokenized = tokenizer.decode(tokenized_list)
        print(untokenized)
        untok_answer = tokenizer.decode(answer_2)

        print('sentence == untokenized', sentence == untokenized)
        print('sentence in untokenized', sentence in untokenized)
        print('str(done_tokens)', str(done_tokens))
        print('done_tokens in tokenized', str(done_tokens) in str_list)
        print('answer_tokens in tokenized', str(answer_tokens) in str_list)
        print('answer_1  ', answer)
        print('answer_2  ', answer_2)
        print('listanswer', tokenized_list[list_ai:list_di])
        print('untok_answer', untok_answer)
        print(list_ai)


def test_small_get_dataset():
    dataset = get_dataset_lambada()
    print(dataset)
    print(dataset.keys())
    print(dataset['train'][0])
    # print(dataset['train'][1])
    # print(dataset['train'][2])


def test_get_lmeval_datasets():
    from thepebbletrail_official.neuron_utils.helpers_models import get_model
    from lm_eval import simple_evaluate, models

    model, tokenizer, _ = get_model('ourllm', 'byt5', n_layers=1, width=256)

    tasks = ["lambada_openai", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande", "openbookqa"]
    # tasks = ["lambada_openai"]
    limit = 2
    batch_size = 1

    model = models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer, max_length=512)
    simple_evaluate(model, tasks=tasks, limit=limit, batch_size=batch_size)
    print('finished successfully')


metrics_for_centrality = [
    "mean",
    "std",
    "mean_dx",
    "std_dx",
    "autocorr",
    "rfourier",
]
centrality_types = [
    *metrics_for_centrality,
    *['anti+' + ct for ct in metrics_for_centrality]
]


def format_qa(x):
    prompt = x["instruction"]
    if x["input"]:
        prompt += "\n" + x["input"]
    return {"text": prompt + "\n" + x["output"]}


def take(ds, n):
    return IterableDataset.from_generator(
        lambda: itertools.islice(ds, n)
    )


def materialize_and_save(iterable_ds, path):
    ds = Dataset.from_generator(lambda: (x for x in iterable_ds))
    ds.save_to_disk(path)
    del ds
    gc.collect()


def non_empty(x):
    return x["text"].strip() != ""


def sanitize_text(example):
    text = example.get("text", None)

    if text is None:
        text = ""

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    return {"text": text}


def is_valid(example):
    return isinstance(example.get("text"), str) and len(example["text"]) > 0


def chunk_text_batch(batch):
    texts = batch["text"]
    out = []
    for text in texts:
        for i in range(0, len(text), MAX_CHARS):
            chunk = text[i: i + MAX_CHARS].strip()
            if chunk:
                out.append(chunk)
    return {"text": out}


def load_sharded_dataset(path):
    shards = sorted(glob(os.path.join(path, "shard_*")))
    return concatenate_datasets([load_from_disk(s) for s in shards])


def get_lmmix1_dataset(notes="", cachedir=None):
    data_path = os.path.join(cachedir, "lmmix1")
    if not os.path.exists(data_path):

        SEED = 42
        random.seed(SEED)

        # ------------------------------------------------------------
        # 1) Load datasets
        #    → STREAMING for large ones
        # ------------------------------------------------------------

        # WikiText-103 (small enough, non-streaming)
        wikitext = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split="train"
        )

        # Wikipedia (STREAMING)
        wikipedia = load_dataset(
            "wikipedia",
            "20220301.en",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        # Public-domain books (STREAMING)
        books = load_dataset(
            "spaul25/Chronoberg",
            split="train",
            streaming=True
        )

        # Code (STREAMING, python only)
        code = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        # QA / instruction data (small, non-streaming is OK)
        qa = load_dataset(
            "tatsu-lab/alpaca",
            split="train",
            trust_remote_code=True
        )

        # ------------------------------------------------------------
        # 2) Normalize all datasets to a single "text" column
        # ------------------------------------------------------------

        wikipedia = wikipedia.map(lambda x: {"text": x["text"]})
        books = books.map(
            chunk_text_batch,
            batched=True,
            batch_size=1,  # CRITICAL for memory safety
            remove_columns=books.column_names,
        ).map(sanitize_text).filter(is_valid)
        code = code.map(lambda x: {"text": x["content"]})
        qa = qa.map(format_qa, remove_columns=qa.column_names)

        datasets_dict = {
            "books": books,
            "wikitext": wikitext,
            "wikipedia": wikipedia,
            "code": code,
            "qa": qa,
        }

        datasets_dict = {
            name: ds.map(sanitize_text).filter(is_valid)
            for name, ds in datasets_dict.items()
        }

        base_path = os.path.join(cachedir, "lmmix1_parts")
        os.makedirs(base_path, exist_ok=True)

        for name, ds in datasets_dict.items():
            print(f"{name} dataset:", ds)
            # print(next(iter(ds)))

            n_samples = 50_000 if name in ["books", "qa", "code"] else 100_000
            out_dir = os.path.join(base_path, name)
            os.makedirs(out_dir, exist_ok=True)

            # ds_iter = take(ds.shuffle(seed=SEED), n_samples)
            # ds_iter = ds.shuffle(seed=SEED)
            ds_iter = ds if name == "books" else ds.shuffle(seed=SEED)

            buffer = []
            shard_id = 0
            seen = 0
            for ex in tqdm(ds_iter, desc=f"Sharding {name} dataset"):
                buffer.append(ex)
                seen += 1

                if len(buffer) == SAVE_EVERY:
                    shard_path = os.path.join(out_dir, f"shard_{shard_id:03d}")
                    Dataset.from_list(buffer).save_to_disk(shard_path)
                    print(f"Saved {name} shard {shard_id} ({len(buffer)} samples)")
                    buffer.clear()
                    shard_id += 1
                    gc.collect()

                if seen >= n_samples:
                    break

            # save remainder
            if buffer:
                shard_path = os.path.join(out_dir, f"shard_{shard_id:03d}")
                Dataset.from_list(buffer).save_to_disk(shard_path)
                gc.collect()
                print(f"Saved {name} shard {shard_id} ({len(buffer)} samples)")

            del ds_iter, buffer

        # ------------------------------------------------------------
        # 3) Sample BEFORE materialization (CRITICAL)
        # ------------------------------------------------------------

        # Recommended example counts for 1 GPU
        datasets_list = [
            load_sharded_dataset(os.path.join(base_path, name))
            for name in ["wikitext", "wikipedia", "books", "code", "qa"]
        ]

        # ------------------------------------------------------------
        # 4) Concatenate and materialize to disk
        # ------------------------------------------------------------

        train_iterable = concatenate_datasets(datasets_list)

        # IMPORTANT:
        # convert iterable → normal Dataset so we can save_to_disk
        features = Features({
            "text": Value("string"),
        })
        train_dataset = Dataset.from_generator(
            lambda: (x for x in train_iterable),
            features=features,

        )
        train_dataset = train_dataset.shuffle(seed=SEED)

        # ------------------------------------------------------------
        # Sanity check
        # ------------------------------------------------------------

        print(train_dataset)
        print(train_dataset[0]["text"][:300])

        # ------------------------------------------------------------
        # Validation and Test from PTB
        # ------------------------------------------------------------

        ptb = get_dataset_ptb(notes=notes)
        val_dataset = ptb["validation"]
        test_dataset = ptb["test"]

        dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        })

        dataset.save_to_disk(data_path)

    dataset = DatasetDict.load_from_disk(data_path)
    return dataset


def get_open_llm_leaderboard(cachedir):
    from transformers import AutoConfig, AutoModelForCausalLM
    from thepebbletrail_official.dataset_utils.model_expressivity import get_expressivities
    from thepebbletrail_official.neuron_utils.helpers_models import model_ids

    batch_size = 4  # 4
    initializations = 20  # 40
    time_steps = 10  # 7

    skip_top = 0  # 10
    test_samples, val_samples = 0, 0  # 20, 20
    n_models = 100
    # load this dataset open-llm-leaderboard/results
    ds = load_dataset('open-llm-leaderboard/contents')['train']
    print(ds)

    # sort by Average column
    df = ds.to_pandas()

    av_col = [col for col in df.columns if 'Average' in col][0]
    par_col = [col for col in df.columns if 'Params' in col][0]

    df = df.sort_values(by=av_col, ascending=False)[skip_top:]
    df_test = df[:test_samples]
    df_val = df[test_samples:val_samples + test_samples]
    df_train = df[val_samples + test_samples:][:n_models]
    df = df_train
    print(df.head().to_string())

    results_path = os.path.join(
        cachedir, 'open-llm-leaderboard',
        f'expressivity_results_b{batch_size}_ins{initializations}_ts{time_steps}.txt'
    )

    for i in range(n_models):
        try:
            print('-' * 50)
            print(f"Model {i + 1}:")
            print(f"    Name:    {df.iloc[i]['fullname']}")
            print(f"    Average: {df.iloc[i][av_col]}")
            print(f"    Params:  {df.iloc[i][par_col]}")

            config = AutoConfig.from_pretrained(df.iloc[i]['fullname'])
            original_num_layers = config.num_hidden_layers
            config.num_hidden_layers = 1

            def generate_model(*args):
                return AutoModelForCausalLM.from_config(config)

            results = get_expressivities(
                batch_size=batch_size,  # 4
                initializations=initializations,  # 40
                time_steps=time_steps,  # 7
                generate_model=generate_model,
            )
            results['original_num_layers'] = original_num_layers
            results['fullname'] = df.iloc[i]['fullname']
            results['Average'] = df.iloc[i][av_col]
            results['Params'] = df.iloc[i][par_col]
            print(results)

            with open(results_path, 'a') as f:
                f.write(results)
                f.write('\n')

        except Exception as e:
            print(f"Error: {e}")
            continue

    for _, fullname in model_ids.items():
        try:

            config = AutoConfig.from_pretrained(fullname)
            original_num_layers = config.num_hidden_layers
            config.num_hidden_layers = 1

            def generate_model(*args):
                return AutoModelForCausalLM.from_config(config)

            results = get_expressivities(
                batch_size=batch_size,  # 4
                initializations=initializations,  # 40
                time_steps=time_steps,  # 7
                generate_model=generate_model,
            )
            results['original_num_layers'] = original_num_layers
            results['fullname'] = fullname
            results['Average'] = -1

            total_params = sum([p.numel() for _, p in generate_model().named_parameters()])
            results['Params'] = total_params
            print(results)

            with open(results_path, 'a') as f:
                f.write(results)
                f.write('\n')

        except Exception as e:
            print(f"Error: {e}")
            continue


def test_dataset_line_stats():
    """Show line counts and average line length for ptb, wiki103, and llmmix1."""
    print("\n" + "=" * 60)
    print("Dataset line statistics (ptb, wiki103, llmmix1)")
    print("=" * 60)

    for name, loader in [
        ("ptb", get_dataset_ptb),
        ("wiki103", get_dataset_wiki103),
        ("llmmix1", get_lmmix1_dataset),
    ]:
        print(f"\n--- {name} ---")
        ds = loader()
        total_lines = 0
        lengths = []

        sample_llmmix1 = name == "llmmix1" and "train" in ds and len(ds["train"]) > 1000

        for split in ds.keys():
            n = len(ds[split])
            total_lines += n
            if sample_llmmix1 and split == "train":
                indices = random.sample(range(n), 1000)
                lengths.extend([len(ds[split][i]["text"]) for i in indices])
            elif not sample_llmmix1:
                lengths.extend([len(ds[split][i]["text"]) for i in range(n)])

        avg_len = sum(lengths) / len(lengths) if lengths else 0
        est_chars = total_lines * avg_len
        print(f"  Total lines: {total_lines:,}")
        if sample_llmmix1:
            print(f"  Avg line length (chars): {avg_len:.1f} (estimated from 1000 sampled train lines)")
        else:
            print(f"  Avg line length (chars): {avg_len:.1f}")
        print(f"  Est. dataset size (chars): {est_chars:,.0f}")

    print("\n" + "=" * 60)


def evaluation(
        model, tokenizer, dataset, dataset_name, eval_split,
        batch_size=1, seed=42, output_dir=None, notes='',
        compute_metrics=None, collator=None,
):
    from trl import SFTConfig
    # from thepebbletrail_official.dataset_utils.helpers_datasets import get_metrics
    from pyaromatics.hf_tools.trainers import PlusTrainer

    config_args = {
        'output_dir': output_dir,
        'per_device_train_batch_size': batch_size,
        'per_device_eval_batch_size': batch_size,
        'logging_strategy': "no",
        'seed': seed,
        'max_steps': len(dataset[eval_split]) // batch_size,
        'batch_eval_metrics': True,
        # 'auto_find_batch_size': True,
        'auto_find_batch_size': False,
        'dataset_text_field': "text",
        'max_length': 60_000,
        'fp16': True,
    }

    assert config_args['auto_find_batch_size'] is False, "auto_find_batch_size has to be set to False to avoid noise."

    if not collator is None:
        config_args['dataset_kwargs'] = {}
        config_args['dataset_kwargs']['skip_prepare_dataset'] = True
        config_args['remove_unused_columns'] = False

    if 'maxlen' in notes or 'dolma' in dataset_name:
        config_args['max_length'] = str2val(notes, 'maxlen', default=256, output_type=int)

    if 'packing' in notes:
        config_args['packing'] = True

    eval_args = SFTConfig(**config_args)
    if not hasattr(eval_args, "past_index"):
        eval_args.past_index = -1

    compute_metrics = None
    model.eval()
    trainer_kwargs = {
        'model': model,
        'processing_class': tokenizer,
        'args': eval_args,
        'train_dataset': dataset["train"],
        'eval_dataset': dataset[eval_split],
        'compute_metrics': compute_metrics,
        'data_collator': collator,
    }

    validator = PlusTrainer(**trainer_kwargs)
    eval_output = validator.evaluate()
    # print('eval_output', eval_output)
    return eval_output


def evaluation_lmeval(
        model, tokenizer, notes='',
        cachepath=None
):
    results = {}
    # LM Eval evaluation loop
    print('\n\n' + '=' * 50)
    print('Running LM Eval on standard benchmarks')
    print('=' * 50)
    try:
        # Ensure we use the HF cache directory for datasets
        # This helps with offline loading if datasets were previously downloaded
        if 'HF_DATASETS_CACHE' not in os.environ and cachepath:
            os.environ['HF_DATASETS_CACHE'] = cachepath
            os.environ['HF_HOME'] = cachepath

        print(f'Using HF cache directory: {os.environ.get("HF_DATASETS_CACHE", "default")}')

        # Define tasks: wikitext, LAMBADA, knowledge/QA + reading comprehension
        tasks = [
            # "wikitext",  # wikitext 2 perplexity (lower is better)
            "lambada_openai",  # LAMBADA
            "boolq",  # BoolQ
            "piqa",  # PIQA
            "hellaswag",  # HellaSwag
            "winogrande",  # Winogrande
            "arc_easy",  # ARC-e
            "arc_challenge",  # ARC-c
            "openbookqa",  # OBQA
        ]

        if 'qaretrieval' in notes:
            tasks += [
                # QA / reading comprehension (lm_eval 0.4.x task keys: squad_completion, squadv2)
                "squad_completion",  # SQuAD-style completion (was "squad" in older harness)
                "squadv2",  # SQuAD v2.0 (was "squad2" in older harness)
                "triviaqa",  # TriviaQA
                "nq_open",  # Natural Questions (open-domain)
                "drop",  # DROP (discrete reasoning)
            ]

        print(f'Evaluating on tasks: {tasks}')

        # Wrap model for lm_eval (registry works across lm_eval versions)
        try:
            hf_model_cls = get_lm_eval_model("hf")
        except Exception:
            hf_model_cls = HFLM
        # Batched forward passes for loglikelihood / scoring (MC tasks). Default HFLM batch_size=1 is slow.
        # generate_until still does autoregressive decoding; batching helps less there but can still batch prompts.
        eval_model = hf_model_cls(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=8,
            max_batch_size=64,
        )

        limit = 2 if 'onlytesting' in notes else None
        lm_eval_results = simple_evaluate(eval_model, tasks=tasks, limit=limit)

        highlights = {}
        for task, res in lm_eval_results['results'].items():
            print(f'Processing results for {task}...')
            print(f'  Available metrics: {list(res.keys())}')
            task_highlights = None
            if 'perplexity,none' in res:
                task_highlights = {'perplexity': res['perplexity,none']}
            elif 'acc,none' in res:
                task_highlights = {'accuracy': res['acc,none']}
            elif 'word_perplexity,none' in res:
                task_highlights = {
                    'word_perplexity': res['word_perplexity,none'],
                    'byte_perplexity': res.get('byte_perplexity,none', 'N/A')
                }
            else:
                # QA tasks: exact_match, f1, em (metric keys may use ,none suffix)
                qa_metrics = {}
                for em_key in ['exact_match,none', 'exact_match', 'em,none', 'em', 'exact_match,remove_whitespace']:
                    if em_key in res:
                        qa_metrics['exact_match'] = res[em_key]
                        break
                for f1_key in ['f1,none', 'f1']:
                    if f1_key in res:
                        qa_metrics['f1'] = res[f1_key]
                        break
                if qa_metrics:
                    task_highlights = qa_metrics
            # Fallback: capture any numeric metrics we didn't match (e.g. task-specific keys)
            if task_highlights is None and res:
                fallback = {k: v for k, v in res.items()
                            if isinstance(v, (int, float, np.number))}
                if fallback:
                    task_highlights = fallback
            if task_highlights is not None:
                highlights[task] = task_highlights

        # remove all keys except results from lm_eval_results
        lm_eval_results = {'results': lm_eval_results['results']}
        results['lm_eval_results'] = lm_eval_results

        for task, res in highlights.items():
            for metric, value in res.items():
                print(f'  {task} - {metric}: {value}')
                results[f'{task}_{metric}'] = value

        print('\nLM Eval Results:')
        print('=' * 50)
        print(json.dumps(highlights, indent=2, cls=NumpyEncoder))
        print('=' * 50)

    except Exception as e:
        error_str = str(e)
        print(f'\nError during lm_eval: {e}')
        import traceback
        traceback.print_exc()

        results['lm_eval_error'] = str(e)

    return results
