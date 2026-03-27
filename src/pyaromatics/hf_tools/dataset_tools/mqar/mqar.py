"""
Multi-query associative recall (MQAR) synthetic data as in Zoology
(Arora, Eyuboglu, et al., ICLR 2024).

Logic matches ``zoology.data.multiquery_ar.multiquery_ar`` from
https://github.com/HazyResearch/zoology (no runtime dependency on zoology).
"""

import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
from datasets import Dataset, DatasetDict


def multiquery_ar_numpy(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Generate MQAR inputs and labels (integer token ids).

    Returns
    -------
    inputs : ndarray, shape (num_examples, input_seq_len)
    labels : ndarray, same shape; use -100 for positions with no prediction target
    slices : metadata dict (constant across the batch)
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 * num_passes + num_kv_pairs * 2 <= input_seq_len

    # Match zoology: single numpy seed drives key/value/gap sampling (see multiquery_ar.py).
    np.random.seed(seed)

    context_size = num_kv_pairs * 2 * num_passes

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(
        np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs
    )

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(
        np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs
    )

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    kvs = np.tile(kvs, (1, num_passes))

    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1, dtype=np.float64) ** (power_a - 1.0)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(
        np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs
    )

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels_full = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(
        labels_full, (gaps * 2) + context_size + 1, values=values, axis=1
    )

    inputs = examples[:, :-1].copy()
    labels = labels_full[:, 1:].copy()

    if random_non_queries:
        mask = inputs == 0
        n_replace = int(mask.sum())
        if n_replace:
            # Zoology uses torch.randint here; numpy draws are equivalent for training.
            inputs[mask] = np.random.randint(0, vocab_size, size=n_replace, dtype=np.int64)

    meta: Dict[str, Any] = {
        "num_kv_pairs": num_kv_pairs,
        "input_seq_len": input_seq_len,
        "num_passes": num_passes,
    }
    return inputs, labels, meta


def generate_mqar_hf_dataset(
    vocab_size: int,
    num_samples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
) -> Dataset:
    """Build a :class:`datasets.Dataset` with ``input_ids`` and ``labels`` columns."""
    inputs, labels, _meta = multiquery_ar_numpy(
        vocab_size=vocab_size,
        num_examples=num_samples,
        input_seq_len=input_seq_len,
        seed=seed,
        power_a=power_a,
        num_kv_pairs=num_kv_pairs,
        num_passes=num_passes,
        random_non_queries=random_non_queries,
    )
    return Dataset.from_dict(
        {
            "input_ids": inputs.tolist(),
            "labels": labels.tolist(),
        }
    )


def build_mqar_dataset_dict(
    train_samples: int = 100_000,
    eval_samples: int = 3_000,
    train_seed: int = 42,
    eval_seed: int = 9_001,
    vocab_size: int = 8_192,
    input_seq_len: int = 64,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
    train_split_name: str = "train",
    eval_split_name: str = "validation",
) -> DatasetDict:
    """
    Train/eval splits with disjoint RNG seeds (same convention as Zoology's
    ``prepare_data``: different seeds for train vs test segments).
    """
    train_ds = generate_mqar_hf_dataset(
        vocab_size=vocab_size,
        num_samples=train_samples,
        input_seq_len=input_seq_len,
        seed=train_seed,
        power_a=power_a,
        num_kv_pairs=num_kv_pairs,
        num_passes=num_passes,
        random_non_queries=random_non_queries,
    )
    eval_ds = generate_mqar_hf_dataset(
        vocab_size=vocab_size,
        num_samples=eval_samples,
        input_seq_len=input_seq_len,
        seed=eval_seed,
        power_a=power_a,
        num_kv_pairs=num_kv_pairs,
        num_passes=num_passes,
        random_non_queries=random_non_queries,
    )
    return DatasetDict(
        {train_split_name: train_ds, eval_split_name: eval_ds}
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HF DatasetDict for Zoology MQAR.")
    p.add_argument("--out", type=str, default=None, help="Directory for save_to_disk")
    p.add_argument(
        "--train-samples", type=int, default=100_000, help="Training examples"
    )
    p.add_argument("--eval-samples", type=int, default=3_000, help="Eval examples")
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=9_001)
    p.add_argument("--vocab-size", type=int, default=8_192)
    p.add_argument("--input-seq-len", type=int, default=64)
    p.add_argument("--power-a", type=float, default=0.1)
    p.add_argument("--num-kv-pairs", type=int, default=8)
    p.add_argument("--num-passes", type=int, default=1)
    p.add_argument(
        "--no-random-non-queries",
        action="store_true",
        help="Keep padding zeros instead of random vocab ids (Zoology default is random).",
    )
    p.add_argument(
        "--eval-split-name",
        type=str,
        default="validation",
        help="Name of the eval split (default: validation)",
    )
    p.add_argument(
        "--train-split-name",
        type=str,
        default="train",
        help="Name of the train split",
    )
    p.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="If set, push DatasetDict to the Hub after building",
    )
    p.add_argument(
        "--meta-json",
        type=str,
        default=None,
        help="Optional path to write generation config JSON next to the dataset",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = {
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "vocab_size": args.vocab_size,
        "input_seq_len": args.input_seq_len,
        "power_a": args.power_a,
        "num_kv_pairs": args.num_kv_pairs,
        "num_passes": args.num_passes,
        "random_non_queries": not args.no_random_non_queries,
        "task": "mqar_zoology",
        "source": "https://github.com/HazyResearch/zoology/blob/main/zoology/data/multiquery_ar.py",
    }

    dsd = build_mqar_dataset_dict(
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        vocab_size=args.vocab_size,
        input_seq_len=args.input_seq_len,
        power_a=args.power_a,
        num_kv_pairs=args.num_kv_pairs,
        num_passes=args.num_passes,
        random_non_queries=config["random_non_queries"],
        train_split_name=args.train_split_name,
        eval_split_name=args.eval_split_name,
    )

    print(dsd)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        dsd.save_to_disk(args.out)
        print(f"Saved DatasetDict to {args.out}")
    if args.meta_json:
        with open(args.meta_json, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Wrote config to {args.meta_json}")
    if args.push_to_hub:
        dsd.push_to_hub(args.push_to_hub)
        print(f"Pushed to {args.push_to_hub}")


if __name__ == "__main__":
    main()
