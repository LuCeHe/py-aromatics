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
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm


def _effective_mqar_chunk_size(input_seq_len: int, chunk_size: Optional[int]) -> int:
    """Cap rows per chunk so one chunk's int64 input+labels stay ~under 64 MiB."""
    if chunk_size is None or chunk_size <= 0:
        chunk_size = 2048
    max_rows = max(32, (64 * 1024 * 1024) // (input_seq_len * 16))
    return min(chunk_size, max_rows)


def _default_parallel_workers(
    input_seq_len: int, parallel_workers: Optional[int]
) -> int:
    if parallel_workers is not None and parallel_workers > 0:
        return parallel_workers
    if input_seq_len >= 1024:
        return min(os.cpu_count() or 1, 8)
    return 1


def _numpy_to_mqar_dataset(inputs: np.ndarray, labels: np.ndarray) -> Dataset:
    return Dataset.from_dict({"input_ids": inputs, "labels": labels})


def _merge_datasets_from_disk(chunk_paths: List[str]) -> Dataset:
    ds = load_from_disk(chunk_paths[0])
    for path in chunk_paths[1:]:
        other = load_from_disk(path)
        ds = concatenate_datasets([ds, other])
        del other
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)
    if len(chunk_paths) > 1:
        shutil.rmtree(chunk_paths[0], ignore_errors=True)
    return ds


def _mqar_chunk_worker(task: Dict[str, Any]) -> str:
    """Generate one MQAR chunk and write it to disk (picklable for ProcessPoolExecutor)."""
    start = task["start"]
    bs = task["bs"]
    chunk_path = task["chunk_path"]
    inp, lab, _ = multiquery_ar_numpy(
        vocab_size=task["vocab_size"],
        num_examples=bs,
        input_seq_len=task["input_seq_len"],
        seed=task["seed"],
        power_a=task["power_a"],
        num_kv_pairs=task["num_kv_pairs"],
        num_passes=task["num_passes"],
        random_non_queries=task["random_non_queries"],
        zoology_shift_labels=task["zoology_shift_labels"],
        row_offset=start,
        independent_rows=task["independent_rows"],
        rng=task.get("rng"),
    )
    _numpy_to_mqar_dataset(inp, lab).save_to_disk(chunk_path)
    del inp, lab
    return chunk_path


def mqar_labels_zoology_to_trl_aligned(labels_stored: np.ndarray) -> np.ndarray:
    """
    Convert MQAR labels from the on-disk Zoology layout to TRL/HF-friendly layout.

    Generation uses ``labels = labels_full[:, 1:]`` (Zoology). TRL's causal LM loss
    also applies a one-step shift (``shift_logits`` vs ``shift_labels``), so targets
    can be misaligned unless the dataset uses the alternate slice
    ``labels = labels_full[:, :-1]`` instead.

    This maps stored rows **without** re-reading ``labels_full``: it is equivalent to
    reconstructing ``labels_full = concat(-100, labels_stored)`` (first column all
    ignore) and then taking ``labels_full[:, :-1]``.
    """
    labels_stored = np.asarray(labels_stored, dtype=np.int64)
    if labels_stored.ndim != 2:
        raise ValueError(f"expected 2D labels array, got shape {labels_stored.shape}")
    pad = np.full((labels_stored.shape[0], 1), -100, dtype=np.int64)
    return np.concatenate([pad, labels_stored[:, :-1]], axis=1)


def multiquery_ar_numpy(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 64,
    num_passes: int = 1,
    random_non_queries: bool = True,
    rng: Optional[np.random.RandomState] = None,
    zoology_shift_labels: bool = True,
    row_offset: int = 0,
    independent_rows: bool = False,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Generate MQAR inputs and labels (integer token ids).

    If ``rng`` is provided, it is used for all draws (and ``seed`` is ignored); this
    allows chunked generation that matches one full batch when the RNG state is
    advanced in the same order as row-major ``apply_along_axis``.

    ``zoology_shift_labels`` (default True): if True, ``labels = labels_full[:, 1:]``
    as in Zoology. If False, ``labels = labels_full[:, :-1]`` for alignment with
    TRL/HF causal LM without applying :func:`mqar_labels_zoology_to_trl_aligned` at load time.

    Returns
    -------
    inputs : ndarray, shape (num_examples, input_seq_len)
    labels : ndarray, same shape; use -100 for positions with no prediction target
    slices : metadata dict (constant across the batch)
    """
    if independent_rows:
        inputs = np.empty((num_examples, input_seq_len), dtype=np.int64)
        labels = np.empty((num_examples, input_seq_len), dtype=np.int64)
        meta: Dict[str, Any] = {}
        for i in range(num_examples):
            inp, lab, meta = multiquery_ar_numpy(
                vocab_size=vocab_size,
                num_examples=1,
                input_seq_len=input_seq_len,
                seed=seed,
                power_a=power_a,
                num_kv_pairs=num_kv_pairs,
                num_passes=num_passes,
                random_non_queries=random_non_queries,
                rng=np.random.RandomState(seed + row_offset + i),
                zoology_shift_labels=zoology_shift_labels,
                independent_rows=False,
            )
            inputs[i] = inp[0]
            labels[i] = lab[0]
        return inputs, labels, meta

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
    # Zoology uses labels_full[:, 1:]. Use labels_full[:, :-1] when training with
    # TRL/SFT so the HF causal shift matches a single temporal alignment (see doc
    # on mqar_labels_zoology_to_trl_aligned).
    if zoology_shift_labels:
        labels = labels_full[:, 1:].copy()
    else:
        labels = labels_full[:, :-1].copy()

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
    num_kv_pairs: int = 64,
    num_passes: int = 1,
    random_non_queries: bool = True,
    chunk_size: Optional[int] = 2_048,
    show_progress: bool = True,
    split_desc: str = "",
    zoology_shift_labels: bool = True,
    parallel_workers: Optional[int] = None,
    independent_rows: Optional[bool] = None,
) -> Dataset:
    """Build a :class:`datasets.Dataset` with ``input_ids`` and ``labels`` columns."""
    desc = f"MQAR {split_desc}".strip() if split_desc else "MQAR"
    chunk_size = _effective_mqar_chunk_size(input_seq_len, chunk_size)
    workers = _default_parallel_workers(input_seq_len, parallel_workers)
    if independent_rows is None:
        independent_rows = workers > 1

    use_chunks = (
        chunk_size is not None
        and chunk_size > 0
        and num_samples > chunk_size
    )
    if use_chunks:
        tmp_dir = tempfile.mkdtemp(prefix="mqar_chunks_")
        n_chunks = (num_samples + chunk_size - 1) // chunk_size
        chunk_paths: List[Optional[str]] = [None] * n_chunks
        chunk_tasks: List[Dict[str, Any]] = []
        for chunk_idx, start in enumerate(range(0, num_samples, chunk_size)):
            bs = min(chunk_size, num_samples - start)
            chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_idx:05d}")
            chunk_tasks.append(
                {
                    "start": start,
                    "bs": bs,
                    "chunk_path": chunk_path,
                    "chunk_idx": chunk_idx,
                    "vocab_size": vocab_size,
                    "input_seq_len": input_seq_len,
                    "seed": seed,
                    "power_a": power_a,
                    "num_kv_pairs": num_kv_pairs,
                    "num_passes": num_passes,
                    "random_non_queries": random_non_queries,
                    "zoology_shift_labels": zoology_shift_labels,
                    "independent_rows": independent_rows,
                }
            )

        if workers > 1 and independent_rows:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_mqar_chunk_worker, task): task["chunk_idx"]
                    for task in chunk_tasks
                }
                it = as_completed(futures)
                if show_progress:
                    it = tqdm(
                        it,
                        total=n_chunks,
                        desc=f"{desc} (parallel chunks, {workers} workers)",
                        unit="chunk",
                        leave=True,
                    )
                for fut in it:
                    chunk_idx = futures[fut]
                    chunk_paths[chunk_idx] = fut.result()
        else:
            rng = None if independent_rows else np.random.RandomState(seed)
            it = chunk_tasks
            if show_progress:
                it = tqdm(
                    chunk_tasks,
                    total=n_chunks,
                    desc=f"{desc} (chunks)",
                    unit="chunk",
                    leave=True,
                )
            for task in it:
                task["rng"] = rng
                chunk_paths[task["chunk_idx"]] = _mqar_chunk_worker(task)
                gc.collect()

        ordered_paths = [p for p in chunk_paths if p is not None]
        try:
            out = _merge_datasets_from_disk(ordered_paths)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return out

    inputs, labels, _ = multiquery_ar_numpy(
        vocab_size=vocab_size,
        num_examples=num_samples,
        input_seq_len=input_seq_len,
        seed=seed,
        power_a=power_a,
        num_kv_pairs=num_kv_pairs,
        num_passes=num_passes,
        random_non_queries=random_non_queries,
        zoology_shift_labels=zoology_shift_labels,
        independent_rows=independent_rows,
    )
    return _numpy_to_mqar_dataset(inputs, labels)


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
    num_kv_pairs: int = 64,
    num_passes: int = 1,
    random_non_queries: bool = True,
    train_split_name: str = "train",
    eval_split_name: str = "validation",
    test_split_name: str = "test",
    chunk_size: Optional[int] = 2_048,
    show_progress: bool = True,
    zoology_shift_labels: bool = True,
    parallel_workers: Optional[int] = None,
    independent_rows: Optional[bool] = None,
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
            zoology_shift_labels=zoology_shift_labels,
            parallel_workers=parallel_workers,
            independent_rows=independent_rows,
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
    p.add_argument("--num-kv-pairs", type=int, default=64)
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
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="CPU workers for chunk generation (0 = auto: all CPUs for seq>=1024, else 1).",
    )
    p.add_argument(
        "--sequential-row-seeds",
        action="store_true",
        help="Use legacy sequential RNG across chunks (not parallel-safe).",
    )
    p.add_argument(
        "--trl-aligned-labels",
        action="store_true",
        help="Use labels = labels_full[:, :-1] instead of Zoology's labels_full[:, 1:] "
        "(better match for TRL/HF causal loss without remapping on load).",
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
        "zoology_shift_labels": not args.trl_aligned_labels,
    }

    chunk_size = None if args.chunk_size <= 0 else args.chunk_size
    parallel_workers = None if args.parallel_workers <= 0 else args.parallel_workers
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
        zoology_shift_labels=not args.trl_aligned_labels,
        parallel_workers=parallel_workers,
        independent_rows=False if args.sequential_row_seeds else None,
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
