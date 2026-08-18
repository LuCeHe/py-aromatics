"""
RegBench-style in-context language learning (Akyürek et al., 2024; seq_icl).

Each example concatenates several strings sampled from one random DFA/PFA,
separated by a separator token. Train/val/test use **disjoint** automata.
No runtime dependency on seq_icl / pythomata.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from datasets import Dataset, DatasetDict

PAD_ID = 0
SEP_ID = 1
IGNORE = -100
# Token ids 2..vocab_size-1 are alphabet symbols.


class _SimpleDFA:
    def __init__(self, transitions: list[dict[int, int]], rng: np.random.Generator):
        self.transitions = transitions
        self.rng = rng

    def sample(self, length: int) -> list[int]:
        node = 0
        word = []
        for _ in range(int(length)):
            outgoing = list(self.transitions[node].keys())
            if not outgoing:
                break
            sym = int(self.rng.choice(outgoing))
            word.append(sym)
            node = self.transitions[node][sym]
        return word


def _sample_dfa(
    rng: np.random.Generator,
    *,
    num_nodes: int,
    alphabet: np.ndarray,
    max_outgoing: int,
) -> _SimpleDFA:
    transitions = [{} for _ in range(num_nodes)]
    for node in range(num_nodes):
        n_out = int(rng.integers(1, max(2, max_outgoing + 1)))
        n_out = min(n_out, len(alphabet), max(1, num_nodes - 1))
        symbols = rng.choice(alphabet, size=n_out, replace=False)
        others = [n for n in range(num_nodes) if n != node] or [node]
        dests = rng.choice(others, size=n_out, replace=len(others) < n_out)
        transitions[node] = {int(s): int(d) for s, d in zip(symbols, dests)}
    dfa_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
    return _SimpleDFA(transitions, dfa_rng)


def _encode_example(
    dfa: _SimpleDFA,
    rng: np.random.Generator,
    *,
    n_strings: int,
    max_len_per: int,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokens: list[int] = []
    for i in range(int(n_strings)):
        length = int(rng.integers(1, max(2, max_len_per + 1)))
        tokens.extend(dfa.sample(length))
        if i < n_strings - 1:
            tokens.append(SEP_ID)
    tokens = tokens[:seq_len]
    x = np.array(tokens, dtype=np.int64)
    y = np.concatenate([x[1:], np.array([IGNORE], dtype=np.int64)])
    y[x == SEP_ID] = IGNORE
    out_x = np.full(seq_len, PAD_ID, dtype=np.int64)
    out_y = np.full(seq_len, IGNORE, dtype=np.int64)
    n = min(seq_len, x.shape[0])
    out_x[:n] = x[:n]
    out_y[:n] = y[:n]
    return out_x, out_y


def generate_regbench_split(
    *,
    num_examples: int,
    num_dfas: int,
    seed: int,
    vocab_size: int,
    seq_len: int,
    max_num_nodes: int = 8,
    min_in_context: int = 10,
    max_in_context: int = 20,
    max_outgoing_edges: int = 4,
    max_len_per_example: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    alphabet = np.arange(2, int(vocab_size))
    if alphabet.size < 2:
        raise ValueError("regbench vocab_size must be >= 4 (pad, sep, ≥2 symbols)")
    dfas = []
    attempts = 0
    while len(dfas) < int(num_dfas) and attempts < int(num_dfas) * 20:
        attempts += 1
        n_nodes = int(rng.integers(max(2, max_outgoing_edges), max(3, max_num_nodes + 1)))
        n_alpha = int(rng.integers(max(2, max_outgoing_edges), alphabet.size + 1))
        alpha = rng.choice(alphabet, size=min(n_alpha, alphabet.size), replace=False)
        dfas.append(
            _sample_dfa(
                rng,
                num_nodes=n_nodes,
                alphabet=alpha,
                max_outgoing=max_outgoing_edges,
            )
        )
    xs, ys = [], []
    for i in range(int(num_examples)):
        dfa = dfas[i % len(dfas)]
        n_str = int(rng.integers(min_in_context, max_in_context + 1))
        x, y = _encode_example(
            dfa,
            rng,
            n_strings=n_str,
            max_len_per=max_len_per_example,
            seq_len=seq_len,
        )
        xs.append(x)
        ys.append(y)
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def build_regbench_dataset_dict(
    *,
    train_samples: int = 12_800,
    eval_samples: int = 1_280,
    test_samples: int = 1_280,
    train_seed: int = 0,
    eval_seed: int = 1,
    test_seed: int = 2,
    vocab_size: int = 16,
    seq_len: int = 256,
    train_dfas: int = 1_000,
    eval_dfas: int = 200,
    test_dfas: int = 200,
    **kwargs,
) -> DatasetDict:
    splits = {}
    for name, n, seed, n_dfa in (
        ("train", train_samples, train_seed, train_dfas),
        ("validation", eval_samples, eval_seed, eval_dfas),
        ("test", test_samples, test_seed, test_dfas),
    ):
        x, y = generate_regbench_split(
            num_examples=n,
            num_dfas=n_dfa,
            seed=seed,
            vocab_size=vocab_size,
            seq_len=seq_len,
            **kwargs,
        )
        splits[name] = Dataset.from_dict({"input_ids": x.tolist(), "labels": y.tolist()})
    return DatasetDict(splits)


def regbench_cache_digest(cache_key: dict) -> str:
    return hashlib.sha256(
        json.dumps(cache_key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
