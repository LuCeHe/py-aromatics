"""
Mechanistic Architecture Design (MAD) synthetics (Poli, Thomas, et al., 2024).

Generators follow ``mad-lab`` (https://github.com/athms/mad-lab): each of the six
tasks is a **separate** train/eval dataset, not one model tested on all six.
No runtime dependency on mad-lab.
"""

from __future__ import annotations

import hashlib
import json
from itertools import permutations
from typing import Callable

import numpy as np
from datasets import Dataset, DatasetDict

IGNORE = -100

MAD_TASK_ALIASES = {
    "mad_in_context_recall": "in-context-recall",
    "mad_icr": "in-context-recall",
    "mad_fuzzy_recall": "fuzzy-in-context-recall",
    "mad_noisy_recall": "noisy-in-context-recall",
    "mad_memorize": "memorization",
    "mad_compress": "compression",
    "mad_selective_copy": "selective-copying",
}

# mad-lab ``configs/tasks/*.yml`` baseline settings.
MAD_TASK_DEFAULTS = {
    "in-context-recall": dict(vocab_size=16, seq_len=128, multi_query=True),
    "fuzzy-in-context-recall": dict(
        vocab_size=16, seq_len=128, multi_query=True, k_motif_size=3, v_motif_size=3,
    ),
    "noisy-in-context-recall": dict(
        vocab_size=32, seq_len=128, multi_query=True, noise_vocab_size=16, frac_noise=0.2,
    ),
    "memorization": dict(vocab_size=256, seq_len=32),
    "compression": dict(vocab_size=16, seq_len=32),
    "selective-copying": dict(vocab_size=16, seq_len=256, num_tokens_to_copy=16),
}


def parse_mad_dataset_name(dataset_name: str) -> tuple[str, int | None]:
    """Return ``(mad-lab task id, optional seq_len suffix)``."""
    raw = str(dataset_name).strip().lower().replace("-", "_")
    seq_len = None
    digits = ""
    while raw and raw[-1].isdigit():
        digits = raw[-1] + digits
        raw = raw[:-1]
    if digits:
        seq_len = int(digits)
        raw = raw.rstrip("_")
    if raw not in MAD_TASK_ALIASES:
        raise ValueError(
            f"Unknown MAD dataset {dataset_name!r}. Use one of: "
            + ", ".join(sorted(MAD_TASK_ALIASES))
        )
    return MAD_TASK_ALIASES[raw], seq_len


def _pad_to(inputs: np.ndarray, targets: np.ndarray, seq_len: int, pad_id: int = 0):
    inputs = np.asarray(inputs, dtype=np.int64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    n = min(int(inputs.shape[0]), int(targets.shape[0]), seq_len)
    out_x = np.full(seq_len, pad_id, dtype=np.int64)
    out_y = np.full(seq_len, IGNORE, dtype=np.int64)
    out_x[:n] = inputs[:n]
    out_y[:n] = targets[:n]
    return out_x, out_y


def _vocab_perms(vocab, motif_size: int, rng: np.random.Generator):
    values = list(permutations(list(vocab), int(motif_size)))
    rng.shuffle(values)
    return values


def generate_in_context_recall(
    *,
    vocab_size: int,
    seq_len: int,
    rng: np.random.Generator,
    is_training: bool,
    multi_query: bool = True,
    noise_vocab_size: int = 0,
    frac_noise: float = 0.0,
    **_kw,
):
    copy_prefix = vocab_size - 1
    non_special = vocab_size - (1 if not multi_query else 0) - noise_vocab_size
    key_vocab = np.arange(non_special // 2)
    value_vocab = np.arange(non_special // 2, non_special)
    noise_vocab = np.arange(non_special, non_special + noise_vocab_size) if frac_noise > 0 else None
    kv_map: dict[int, int] = {}
    inputs, targets = [], []
    keys_presented: dict[int, int] = {}
    seq_len = seq_len if seq_len % 2 == 0 else seq_len + 1
    num_kv = seq_len // 2
    not_noise_idx = int(rng.integers(0, num_kv))
    for i in range(num_kv - 1):
        is_noise = (
            bool(rng.random() < frac_noise)
            if i != not_noise_idx and frac_noise > 0
            else False
        )
        if is_noise and noise_vocab is not None:
            noise = rng.choice(noise_vocab, size=2, replace=True)
            inputs.extend(list(noise))
            targets.extend([IGNORE, IGNORE])
            continue
        k = int(rng.choice(key_vocab))
        if k not in kv_map:
            kv_map[k] = int(rng.choice(value_vocab))
        v = kv_map[k]
        inputs.extend([k, v])
        targets.append(IGNORE)
        if k not in keys_presented:
            targets.append(IGNORE)
        else:
            targets.append(v if multi_query else IGNORE)
        keys_presented[k] = v
    k_probe = int(rng.choice(list(keys_presented.keys()))) if keys_presented else int(rng.choice(key_vocab))
    v_probe = keys_presented.get(k_probe, int(rng.choice(value_vocab)))
    if not multi_query:
        inputs.extend([copy_prefix, k_probe, v_probe])
        targets.extend([IGNORE, IGNORE, v_probe])
    else:
        inputs.extend([k_probe, v_probe])
        targets.extend([IGNORE, v_probe])
    x = np.asarray(inputs, dtype=np.int64)
    y = np.asarray(targets, dtype=np.int64)
    if is_training:
        y = np.concatenate([x[1:], np.array([IGNORE], dtype=np.int64)])[: x.shape[0]]
    return _pad_to(x, y, seq_len)


def generate_fuzzy_in_context_recall(
    *,
    vocab_size: int,
    seq_len: int,
    rng: np.random.Generator,
    is_training: bool,
    multi_query: bool = True,
    k_motif_size: int = 3,
    v_motif_size: int = 3,
    **_kw,
):
    copy_prefix = vocab_size - 1
    pad_token = vocab_size - 2 if not multi_query else vocab_size - 1
    non_special = vocab_size - (2 if not multi_query else 1)
    key_vocab = np.arange(non_special // 2)
    value_vocab = np.arange(non_special // 2, non_special)
    keys = {
        s: _vocab_perms(key_vocab, s, rng)
        for s in range(1, k_motif_size + 1)
    }
    values = {
        s: _vocab_perms(value_vocab, s, rng)
        for s in range(1, v_motif_size + 1)
    }
    k_probe_size = int(rng.choice(list(keys.keys()))) if is_training else k_motif_size
    v_probe_size = int(rng.choice(list(values.keys())))
    k_probe = tuple(rng.choice(keys[k_probe_size]))
    v_probe = tuple(rng.choice(values[v_probe_size]))
    kv_map: dict[int, dict] = {s: {} for s in range(1, k_motif_size + 1)}
    inputs, targets = [], []
    keys_presented: dict = {}
    probe_added = False
    while len(inputs) < seq_len - (k_motif_size + v_motif_size) - 4:
        if not probe_added and len(inputs) > 4:
            inputs.extend(k_probe)
            inputs.extend(v_probe)
            targets.extend([IGNORE] * (len(k_probe) + len(v_probe)))
            kv_map[k_probe_size][k_probe] = v_probe
            keys_presented[k_probe] = v_probe
            probe_added = True
            continue
        k_size = int(rng.choice(list(keys.keys()))) if is_training else k_motif_size
        v_size = int(rng.choice(list(values.keys())))
        k = tuple(rng.choice(keys[k_size]))
        if k == k_probe:
            v = v_probe
            probe_added = True
        elif k not in kv_map[k_size]:
            v = tuple(rng.choice(values[v_size]))
            kv_map[k_size][k] = v
        else:
            v = kv_map[k_size][k]
        inputs.extend(k)
        inputs.extend(v)
        targets.extend([IGNORE] * len(k))
        if k not in keys_presented:
            targets.extend([IGNORE] * len(v))
        else:
            targets.extend(list(v) if multi_query else [IGNORE] * len(v))
        keys_presented[k] = v
    if not multi_query:
        inputs.extend([copy_prefix])
        inputs.extend(k_probe)
        inputs.extend(v_probe)
        targets.extend([IGNORE] * (1 + len(k_probe)))
        targets.extend(list(v_probe))
    x = np.asarray(inputs, dtype=np.int64)
    y = np.asarray(targets, dtype=np.int64)
    if is_training:
        y = np.concatenate([x[1:], np.array([IGNORE], dtype=np.int64)])[: x.shape[0]]
    return _pad_to(x, y, seq_len, pad_id=int(pad_token))


def generate_memorization(
    *,
    vocab_size: int,
    seq_len: int,
    rng: np.random.Generator,
    kv_map: dict | None = None,
    kv_map_seed: int = 12345,
    **_kw,
):
    insert_token = vocab_size - 1
    non_special = vocab_size - 1
    if kv_map is None:
        kv_rng = np.random.default_rng(kv_map_seed)
        key_vocab = np.arange(non_special // 2)
        value_vocab = np.arange(non_special // 2, non_special)
        keys = list(key_vocab)
        kv_rng.shuffle(keys)
        vals = list(value_vocab)
        kv_rng.shuffle(vals)
        kv_map = {int(k): int(v) for k, v in zip(keys, vals)}
    keys = list(kv_map.keys())
    inputs, targets = [], []
    num_kv = max(1, seq_len // 2)
    for _ in range(num_kv):
        k = int(rng.choice(keys))
        v = int(kv_map[k])
        inputs.extend([k, insert_token])
        targets.extend([IGNORE, v])
    return _pad_to(np.asarray(inputs), np.asarray(targets), seq_len)


def generate_compression(
    *,
    vocab_size: int,
    seq_len: int,
    rng: np.random.Generator,
    **_kw,
):
    compression_token = vocab_size - 1
    vocab = np.arange(vocab_size - 1)
    body = rng.choice(vocab, size=(seq_len - 1,), replace=True)
    inputs = np.concatenate([body, np.array([compression_token], dtype=np.int64)])
    targets = inputs.copy()
    return _pad_to(inputs, targets, seq_len)


def generate_selective_copying(
    *,
    vocab_size: int,
    seq_len: int,
    rng: np.random.Generator,
    num_tokens_to_copy: int = 16,
    **_kw,
):
    copy_token = vocab_size - 1
    blank_token = vocab_size - 2
    vocab = np.arange(vocab_size - 2)
    n_copy = min(int(num_tokens_to_copy), max(1, (seq_len - 3) // 2))
    to_copy = rng.choice(vocab, size=(n_copy,), replace=True)
    n_blank = seq_len - (n_copy * 2) - 1
    inputs = list(to_copy)
    insert_at = rng.integers(0, len(inputs) + 1, size=max(0, n_blank))
    for idx in sorted(insert_at, reverse=True):
        inputs.insert(int(idx), blank_token)
    inputs.append(copy_token)
    inputs.extend([blank_token] * n_copy)
    targets = [IGNORE] * (len(inputs) - n_copy)
    targets.extend(list(to_copy))
    return _pad_to(np.asarray(inputs), np.asarray(targets), seq_len, pad_id=blank_token)


_GENERATORS: dict[str, Callable] = {
    "in-context-recall": generate_in_context_recall,
    "fuzzy-in-context-recall": generate_fuzzy_in_context_recall,
    "noisy-in-context-recall": generate_in_context_recall,
    "memorization": generate_memorization,
    "compression": generate_compression,
    "selective-copying": generate_selective_copying,
}


def generate_mad_split(
    task: str,
    *,
    num_examples: int,
    seed: int,
    is_training: bool,
    vocab_size: int,
    seq_len: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    gen = _GENERATORS[task]
    rng = np.random.default_rng(int(seed))
    xs, ys = [], []
    kw = dict(kwargs)
    if task == "memorization":
        kw.setdefault("kv_map_seed", int(seed))
    for _ in range(int(num_examples)):
        x, y = gen(
            vocab_size=vocab_size,
            seq_len=seq_len,
            rng=rng,
            is_training=is_training,
            **kw,
        )
        xs.append(x)
        ys.append(y)
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def build_mad_dataset_dict(
    task: str,
    *,
    train_samples: int,
    eval_samples: int,
    test_samples: int,
    train_seed: int,
    eval_seed: int,
    test_seed: int,
    vocab_size: int,
    seq_len: int,
    **task_kwargs,
) -> DatasetDict:
    splits = {}
    for name, n, seed, train in (
        ("train", train_samples, train_seed, True),
        ("validation", eval_samples, eval_seed, False),
        ("test", test_samples, test_seed, False),
    ):
        x, y = generate_mad_split(
            task,
            num_examples=n,
            seed=seed,
            is_training=train,
            vocab_size=vocab_size,
            seq_len=seq_len,
            **task_kwargs,
        )
        splits[name] = Dataset.from_dict({"input_ids": x.tolist(), "labels": y.tolist()})
    return DatasetDict(splits)


def mad_cache_digest(cache_key: dict) -> str:
    return hashlib.sha256(
        json.dumps(cache_key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
