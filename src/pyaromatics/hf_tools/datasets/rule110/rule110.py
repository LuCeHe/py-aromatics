# rule110_hf_dataset.py

import random
from datasets import Dataset


# -------------------------------
# Rule 110 core
# -------------------------------

def rule110_step(cells):
    rule = {
        (1, 1, 1): 0,
        (1, 1, 0): 1,
        (1, 0, 1): 1,
        (1, 0, 0): 0,
        (0, 1, 1): 1,
        (0, 1, 0): 1,
        (0, 0, 1): 1,
        (0, 0, 0): 0,
    }

    padded = [0] + cells + [0]
    return [
        rule[tuple(padded[i:i + 3])]
        for i in range(len(cells))
    ]


def generate_rule110(width, steps, init="random"):
    if init == "single":
        cells = [0] * width
        cells[width // 2] = 1
    elif init == "random":
        cells = [random.randint(0, 1) for _ in range(width)]
    else:
        cells = list(init)

    history = [cells]
    for _ in range(steps):
        cells = rule110_step(cells)
        history.append(cells)

    return history


# -------------------------------
# Bit extraction and tokenization
# -------------------------------

def columns_to_bitstrings(history):
    steps = len(history)
    width = len(history[0])

    return [
        ''.join(str(history[t][x]) for t in range(steps))
        for x in range(width)
    ]


def column_bits_to_token_indices(column_bits, vocab_size):
    return [
        int(bits, 2) % vocab_size
        for bits in column_bits
    ]


def rule110_generate_tokens(
    vocab,
    max_length,
    steps,
    init="random"
):
    history = generate_rule110(
        width=max_length,
        steps=steps,
        init=init
    )

    column_bits = columns_to_bitstrings(history)
    indices = column_bits_to_token_indices(
        column_bits,
        vocab_size=len(vocab)
    )

    return [vocab[i] for i in indices]


# -------------------------------
# HF Dataset generator
# -------------------------------

def generate_rule110_dataset(
    vocab,
    num_samples,
    max_length,
    steps,
    init="random",
    seed=None
):
    """
    Returns a Hugging Face Dataset with `num_samples` samples.

    Each sample:
      { "tokens": List[str] }
    """
    if seed is not None:
        random.seed(seed)

    samples = []
    for _ in range(num_samples):
        tokens = rule110_generate_tokens(
            vocab=vocab,
            max_length=max_length,
            steps=steps,
            init=init
        )
        samples.append({"tokens": tokens})

    return Dataset.from_list(samples)


# -------------------------------
# Example usage
# -------------------------------

if __name__ == "__main__":
    vocab = [f"tok{i}" for i in range(256)]

    dataset = generate_rule110_dataset(
        vocab=vocab,
        num_samples=1000,
        max_length=64,
        steps=12,
        init="random",
        # seed=42
    )

    print(dataset)
    print(dataset[0])
