"""
Multi-query associative recall (MQAR) synthetic data as in Zoology
(Arora, Eyuboglu, et al., ICLR 2024).

Logic matches ``zoology.data.multiquery_ar.multiquery_ar`` from
https://github.com/HazyResearch/zoology (no runtime dependency on zoology).
"""

import argparse
import gc
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm


def multiquery_ar_numpy(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Generate MQAR inputs and labels (integer token ids).

    If ``rng`` is provided, it is used for all draws (and ``seed`` is ignored); this
    allows chunked generation that matches one full batch when the RNG state is
    advanced in the same order as row-major ``apply_along_axis``.

    Returns
    -------
    inputs : ndarray, shape (num_examples, input_seq_len)
    labels : ndarray, same shape; use -100 for positions with no prediction target
    slices : metadata dict (constant across the batch)
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    min_len = num_kv_pairs * 2 * num_passes + num_kv_pairs * 2  # == 2 * num_kv_pairs * (num_passes + 1)
    assert min_len <= input_seq_len, (
        f"MQAR needs input_seq_len >= {min_len} for num_kv_pairs={num_kv_pairs}, "
        f"num_passes={num_passes} (got input_seq_len={input_seq_len}). "
        f"Either use a longer name (e.g. mqar64) or lower numkvpairs / numpasses in notes."
    )

    # Match zoology: numpy RNG drives key/value/gap sampling (see multiquery_ar.py).
    if rng is None:
        rng = np.random.RandomState(seed)

    context_size = num_kv_pairs * 2 * num_passes

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(
        lambda row: rng.choice(row, replace=False, size=num_kv_pairs),
        1,
        keys_unshuffled,
    )

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(
        lambda row: rng.choice(row, replace=False, size=num_kv_pairs),
        1,
        values_unshuffled,
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
        lambda row: rng.choice(row, replace=False, p=p, size=num_kv_pairs),
        axis=1,
        arr=x,
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
            inputs[mask] = rng.randint(0, vocab_size, size=n_replace, dtype=np.int64)

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
    chunk_size: Optional[int] = 2_048,
    show_progress: bool = True,
    split_desc: str = "",
) -> Dataset:
    """Build a :class:`datasets.Dataset` with ``input_ids`` and ``labels`` columns."""
    desc = f"MQAR {split_desc}".strip() if split_desc else "MQAR"

    use_chunks = (
        chunk_size is not None
        and chunk_size > 0
        and num_samples > chunk_size
    )
    if use_chunks:
        rng = np.random.RandomState(seed)
        ds_parts = []
        n_chunks = (num_samples + chunk_size - 1) // chunk_size
        it = range(0, num_samples, chunk_size)
        if show_progress:
            it = tqdm(
                it,
                total=n_chunks,
                desc=f"{desc} (chunks)",
                unit="chunk",
                leave=True,
            )
        for start in it:
            bs = min(chunk_size, num_samples - start)
            inp, lab, _ = multiquery_ar_numpy(
                vocab_size=vocab_size,
                num_examples=bs,
                input_seq_len=input_seq_len,
                seed=seed,
                power_a=power_a,
                num_kv_pairs=num_kv_pairs,
                num_passes=num_passes,
                random_non_queries=random_non_queries,
                rng=rng,
            )
            ds_parts.append(
                Dataset.from_dict(
                    {
                        "input_ids": inp.tolist(),
                        "labels": lab.tolist(),
                    }
                )
            )
            del inp, lab
            gc.collect()
        return concatenate_datasets(ds_parts)

    inputs, labels, _ = multiquery_ar_numpy(
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
    test_samples: int = 3_000,
    train_seed: int = 42,
    eval_seed: int = 9_001,
    test_seed: int = 42_001,
    vocab_size: int = 8_192,
    input_seq_len: int = 64,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
    train_split_name: str = "train",
    eval_split_name: str = "validation",
    test_split_name: str = "test",
    chunk_size: Optional[int] = 2_048,
    show_progress: bool = True,
) -> DatasetDict:
    """
    Train / validation / test splits with disjoint RNG seeds (same convention as
    Zoology's ``prepare_data``: different seeds per segment).
    """
    split_specs = [
        (train_split_name, train_samples, train_seed),
        (eval_split_name, eval_samples, eval_seed),
        (test_split_name, test_samples, test_seed),
    ]
    iterator = split_specs
    if show_progress:
        iterator = tqdm(split_specs, desc="MQAR splits", unit="split", total=3)

    splits: Dict[str, Dataset] = {}
    for name, n, sd in iterator:
        splits[name] = generate_mqar_hf_dataset(
            vocab_size=vocab_size,
            num_samples=n,
            input_seq_len=input_seq_len,
            seed=sd,
            power_a=power_a,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
            random_non_queries=random_non_queries,
            chunk_size=chunk_size,
            show_progress=show_progress,
            split_desc=name,
        )
    return DatasetDict(splits)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HF DatasetDict for Zoology MQAR.")
    p.add_argument("--out", type=str, default=None, help="Directory for save_to_disk")
    p.add_argument(
        "--train-samples", type=int, default=100_000, help="Training examples"
    )
    p.add_argument("--eval-samples", type=int, default=3_000, help="Validation examples")
    p.add_argument("--test-samples", type=int, default=3_000, help="Test examples")
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=9_001)
    p.add_argument("--test-seed", type=int, default=42_001)
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
        help="Name of the validation split (default: validation)",
    )
    p.add_argument(
        "--test-split-name",
        type=str,
        default="test",
        help="Name of the test split (default: test)",
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
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during generation",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2_048,
        help="Max rows per chunk for large splits (0 = one shot). Smaller uses less RAM.",
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

    chunk_size = None if args.chunk_size <= 0 else args.chunk_size
    dsd = build_mqar_dataset_dict(
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        test_samples=args.test_samples,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        test_seed=args.test_seed,
        vocab_size=args.vocab_size,
        input_seq_len=args.input_seq_len,
        power_a=args.power_a,
        num_kv_pairs=args.num_kv_pairs,
        num_passes=args.num_passes,
        random_non_queries=config["random_non_queries"],
        train_split_name=args.train_split_name,
        eval_split_name=args.eval_split_name,
        test_split_name=args.test_split_name,
        chunk_size=chunk_size,
        show_progress=not args.no_progress,
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
