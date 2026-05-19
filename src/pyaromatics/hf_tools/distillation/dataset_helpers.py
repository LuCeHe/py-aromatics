"""Dataset helpers for distillation: loading, tokenizing, teacher generations."""

import os, re, random, json, string
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_from_disk, load_dataset

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


def _strip_special_tokens_from_text(text: str, tokenizer: AutoTokenizer) -> str:
    """Remove EOS/pad and other special token strings from decoded text."""
    if not text:
        return text
    to_strip = []
    if tokenizer.eos_token:
        to_strip.append(tokenizer.eos_token)
    if tokenizer.pad_token and tokenizer.pad_token != tokenizer.eos_token:
        to_strip.append(tokenizer.pad_token)
    for special in (tokenizer.additional_special_tokens or []):
        if special and special not in to_strip:
            to_strip.append(special)
    result = text
    for s in to_strip:
        if s:
            result = result.replace(s, "")
    return result.strip()


def _train_val_test_split(
    dataset: Dataset,
    val_split: float | int,
    test_split: float | int,
    split_seed: int,
) -> dict[str, Dataset]:
    """Split dataset into train/validation/test. Val and test are defined; train gets the rest."""
    n = len(dataset)
    use_fractions = 0 < val_split < 1 and 0 < test_split < 1

    if use_fractions:
        test_size = test_split
        val_frac_of_train_val = val_split / (1 - test_split)
    else:
        test_size = int(test_split)
        val_size = int(val_split)
        if test_size + val_size >= n:
            raise ValueError(f"test_split ({test_size}) + val_split ({val_size}) must be < dataset size ({n})")

    split1 = dataset.train_test_split(test_size=test_size, seed=split_seed)
    train_val, test = split1["train"], split1["test"]
    split2 = train_val.train_test_split(
        test_size=val_frac_of_train_val if use_fractions else val_size,
        seed=split_seed,
    )
    train, validation = split2["train"], split2["test"]
    return {"train": train, "validation": validation, "test": test}


def get_mathlib_extracted(val_split: float | int, test_split: float | int, split_seed: int):
    """harrywsanders/mathlib_extracted dataset."""
    print("Loading mathlib_extracted dataset from HuggingFace...")
    dataset = load_dataset("harrywsanders/mathlib_extracted", split="train")

    def concat_example(example):
        statement = example.get("statement") or ""
        tactic = example.get("tactic") or ""
        name = example.get("name") or ""
        return {"text": f"{name}: {statement} {tactic}"}

    dataset = dataset.map(concat_example, remove_columns=dataset.column_names)
    dataset = _train_val_test_split(dataset, val_split, test_split, split_seed)
    print(f"Dataset split: train={len(dataset['train'])}, val={len(dataset['validation'])}, test={len(dataset['test'])}")
    return dataset


def get_reasoning_dataset(val_split: float | int, test_split: float | int, split_seed: int):
    """ankner/gsm8k-CoT dataset."""
    dataset = load_dataset("ankner/gsm8k-CoT")

    def concat_example(example):
        question = example.get("question") or ""
        answer = example.get("response") or ""
        return {"text": f"Question: {question} Response: {answer}"}

    for split in dataset.keys():
        dataset[split] = dataset[split].map(concat_example, remove_columns=dataset[split].column_names)

    dataset = _train_val_test_split(dataset["train"], val_split, test_split, split_seed)
    return dataset


def get_formalizer_dataset(val_split: float | int, test_split: float | int, split_seed: int):
    """FrenzyMath/Herald_proofs dataset."""
    dataset = load_dataset("FrenzyMath/Herald_proofs")
    print(dataset)

    autoformalize_prompt = (
        "Please autoformalize the following natural language problem statement in Lean 4. "
        "Use the following theorem name: {problem_name}\n"
        "The natural language statement is: \n"
        "{nl_theorem_proof}"
        "\n"
        "Think before you provide the lean statement."
    )

    def concat_example(example):
        nl_theorem_proof = "Theorem: " + example["informal_theorem"] + "\n\n" + "Proof: " + example["informal_proof"]
        problem_name = f"theorem_" + "".join(random.choice(string.ascii_letters) for _ in range(10))
        return {"text": autoformalize_prompt.format(problem_name=problem_name, nl_theorem_proof=nl_theorem_proof)}

    for split in dataset.keys():
        dataset[split] = dataset[split].map(concat_example, remove_columns=dataset[split].column_names)

    dataset = _train_val_test_split(dataset["train"], val_split, test_split, split_seed)
    return dataset


def get_dataset(
    dataset_name: str,
    val_split: float | int,
    test_split: float | int,
    split_seed: int,
) -> dict[str, Dataset]:
    """Load dataset and split into train/validation/test."""
    if dataset_name in ["mathlib_extracted", "proving"]:
        return get_mathlib_extracted(val_split, test_split, split_seed)
    elif dataset_name == "reasoning":
        return get_reasoning_dataset(val_split, test_split, split_seed)
    elif dataset_name == "formalizing":
        return get_formalizer_dataset(val_split, test_split, split_seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: mathlib_extracted, proving, reasoning, formalizing")


def tokenize_dataset(
        dataset: Dataset,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        text_column: str = "text",
) -> Dataset:
    """Tokenize dataset for training."""

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        texts = examples.get(text_column, examples.get("formal_statement", [""]))
        if isinstance(texts, str):
            texts = [texts]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )


def build_student_dataset_from_teacher_generations(
        teacher_model: AutoModelForCausalLM,
        teacher_tokenizer: AutoTokenizer,
        student_tokenizer: AutoTokenizer,
        prompt_dataset: Dataset,
        max_new_tokens: int = 256,
        student_max_length: int = 1024,
        num_samples_per_prompt: int = 1,
        gen_temperature: float = 0.7,
) -> Dataset:
    """Use dataset only to prompt the teacher; student sees only teacher generations (in student token space)."""
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    device = next(teacher_model.parameters()).device
    pad_id = teacher_tokenizer.pad_token_id or teacher_tokenizer.eos_token_id

    rows: list[dict[str, Any]] = []
    n = len(prompt_dataset)

    for idx in tqdm(range(n), desc="Teacher generation → student dataset"):
        example = prompt_dataset[idx]
        input_ids = example["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        else:
            input_ids = input_ids.unsqueeze(0).to(device)

        attention_mask = example.get("attention_mask")
        if attention_mask is not None:
            if isinstance(attention_mask, list):
                attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)
            else:
                attention_mask = attention_mask.unsqueeze(0).to(device)
        else:
            attention_mask = None

        for _ in range(num_samples_per_prompt):
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=gen_temperature,
                    pad_token_id=pad_id,
                    eos_token_id=teacher_tokenizer.eos_token_id,
                )
                if attention_mask is not None:
                    gen_kwargs["attention_mask"] = attention_mask
                out = teacher_model.generate(input_ids, **gen_kwargs)
            input_length = input_ids.shape[1]
            generated_ids = out[:, input_length:]
            text = teacher_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            text = _strip_special_tokens_from_text(text, teacher_tokenizer)
            tok = student_tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=student_max_length,
                return_tensors="pt",
            )
            labels = tok["input_ids"][0].clone()
            labels[tok["attention_mask"][0] == 0] = -100
            rows.append({
                "input_ids": tok["input_ids"][0].tolist(),
                "attention_mask": tok["attention_mask"][0].tolist(),
                "labels": labels.tolist(),
            })

    return Dataset.from_list(rows)


def _generated_cache_dir(
    teacher_model_name: str,
    dataset_name: str,
    split: str,
    val_split: float | int,
    test_split: float | int,
    split_seed: int,
    max_samples: int,
    max_new_tokens: int,
    gen_temperature: float,
    num_samples_per_prompt: int,
    seed: int,
    generated_datasets_dir: str,
) -> Path:
    """Unique cache dir for generated text, keyed by teacher, dataset, split, and gen params."""
    t = teacher_model_name.replace("/", "-")
    split_key = f"v{val_split}_t{test_split}_s{split_seed}"
    key = f"{t}_{dataset_name}_{split}_{split_key}_n{max_samples}_gen{max_new_tokens}_t{gen_temperature}_k{num_samples_per_prompt}_seed{seed}"
    return Path(generated_datasets_dir) / key


def _tokenize_texts_to_dataset(
    texts: list[str],
    student_tokenizer: AutoTokenizer,
    student_max_length: int,
) -> Dataset:
    """Tokenize a list of texts into a Dataset with input_ids, attention_mask, labels."""
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    tok = student_tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=student_max_length,
        return_tensors="pt",
    )

    rows: list[dict[str, Any]] = []
    for i in range(len(texts)):
        labels = tok["input_ids"][i].clone()
        labels[tok["attention_mask"][i] == 0] = -100
        rows.append({
            "input_ids": tok["input_ids"][i].tolist(),
            "attention_mask": tok["attention_mask"][i].tolist(),
            "labels": labels.tolist(),
        })
    return Dataset.from_list(rows)


def _count_generations_jsonl(output_path: Path) -> int:
    """Count entries in generations.jsonl without loading full content."""
    if not output_path.exists():
        return 0
    count = 0
    with open(output_path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_generations_jsonl(output_path: Path, expected_total: int = 0) -> list[str]:
    """Read generations.jsonl and return out_texts sorted by (prompt_index, sample_index). Returns [] if incomplete."""
    if not output_path.exists():
        return []
    entries: list[tuple[tuple[int, int], str]] = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out_text = obj.get("out_text", obj.get("text", ""))
                entries.append(((obj["prompt_index"], obj["sample_index"]), out_text))
    entries.sort(key=lambda x: x[0])
    if expected_total > 0 and len(entries) != expected_total:
        return []
    return [t for _, t in entries]


def build_student_dataset_from_teacher_generations_batched(
        teacher_model: AutoModelForCausalLM,
        teacher_tokenizer: AutoTokenizer,
        student_tokenizer: AutoTokenizer,
        prompt_dataset: Dataset,
        max_new_tokens: int = 256,
        student_max_length: int = 1024,
        num_samples_per_prompt: int = 1,
        gen_temperature: float = 0.7,
        batch_size: int = 8,
        dataset_name: str | None = None,
        split: str = "train",
        val_split: float | int = 0.1,
        test_split: float | int = 0.1,
        split_seed: int = 42,
        max_samples: int = -1,
        seed: int = 42,
        force_regenerate: bool = False,
        cache_dir: str | Path | None = None,
        total_gen_chunk_num: int | None = None,
        current_chunk_num: int | None = None,
        skip_generation: bool = False,
        generated_datasets_dir: str | None = None,
) -> Dataset:
    """Batched version with caching. Saves generations to generations.jsonl."""
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    teacher_name = getattr(teacher_tokenizer, "name_or_path", "unknown")
    student_name = getattr(student_tokenizer, "name_or_path", "unknown").replace("/", "-")
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = Path(generated_datasets_dir) / cache_dir if generated_datasets_dir else cache_dir
    else:
        if generated_datasets_dir is None:
            raise ValueError("generated_datasets_dir must be provided when cache_dir is None")
        cache_dir = _generated_cache_dir(
            teacher_model_name=teacher_name,
            dataset_name=dataset_name or "unknown",
            split=split,
            val_split=val_split,
            test_split=test_split,
            split_seed=split_seed,
            max_samples=max_samples,
            max_new_tokens=max_new_tokens,
            gen_temperature=gen_temperature,
            num_samples_per_prompt=num_samples_per_prompt,
            seed=seed,
            generated_datasets_dir=generated_datasets_dir,
        )
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / "generations.jsonl"
    tokenized_dir = cache_dir / f"tokenized_{student_name}_{student_max_length}"
    metadata_path = cache_dir / "metadata.json"

    n_total = len(prompt_dataset)
    if total_gen_chunk_num is not None and current_chunk_num is not None:
        chunk_size = (n_total + total_gen_chunk_num - 1) // total_gen_chunk_num
        start_idx = current_chunk_num * chunk_size
        end_idx = min(start_idx + chunk_size, n_total)
        if start_idx >= end_idx:
            prompt_dataset = prompt_dataset.select(range(0))
            n = 0
        else:
            prompt_dataset = prompt_dataset.select(range(start_idx, end_idx))
            n = len(prompt_dataset)
    else:
        start_idx = 0
        n = n_total

    total_flat = n_total * num_samples_per_prompt
    chunk_flat = n * num_samples_per_prompt

    texts = _load_generations_jsonl(output_path, total_flat)
    if texts and len(texts) == total_flat:
        print(f"Loaded {len(texts)} cached texts from {output_path}")
    elif n == 0:
        return Dataset.from_list([])
    elif skip_generation:
        print(f"Skipping generation for {split}; cache incomplete ({_count_generations_jsonl(output_path)}/{total_flat})")
        return Dataset.from_list([])
    else:
        device = next(teacher_model.parameters()).device
        pad_id = teacher_tokenizer.pad_token_id or teacher_tokenizer.eos_token_id

        flat_inputs: list[tuple[list[int], list[int] | None, int, int]] = []
        for local_idx in range(n):
            global_prompt_idx = start_idx + local_idx
            example = prompt_dataset[local_idx]
            ids = example["input_ids"]
            ids = ids.tolist() if not isinstance(ids, list) else ids
            mask = example.get("attention_mask")
            mask = mask.tolist() if mask is not None and not isinstance(mask, list) else mask
            for si in range(num_samples_per_prompt):
                flat_inputs.append((ids, mask, global_prompt_idx, si))

        already_done: set[tuple[int, int]] = set()
        if output_path.exists() and not force_regenerate:
            with open(output_path) as f:
                if HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            already_done.add((obj["prompt_index"], obj["sample_index"]))
                finally:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            flat_inputs = [(ids, mask, pi, si) for ids, mask, pi, si in flat_inputs if (pi, si) not in already_done]

        torch.manual_seed(seed)
        for start in tqdm(range(0, len(flat_inputs), batch_size), desc="Teacher generation (batched) → text"):
            batch_inputs = flat_inputs[start: start + batch_size]
            max_len = max(len(ids) for ids, _, _, _ in batch_inputs)
            batch_ids = []
            batch_mask = []
            for ids, mask, _, _ in batch_inputs:
                pad_len = max_len - len(ids)
                padded_ids = ids + [pad_id] * pad_len
                batch_ids.append(padded_ids)
                batch_mask.append((mask or [1] * len(ids)) + [0] * pad_len)

            input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch_mask, dtype=torch.long, device=device)

            with torch.no_grad():
                out = teacher_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=gen_temperature,
                    pad_token_id=pad_id,
                    eos_token_id=teacher_tokenizer.eos_token_id,
                )
            input_length = input_ids.shape[1]
            generated_ids = out[:, input_length:]
            batch_out_texts = teacher_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            batch_out_texts = [_strip_special_tokens_from_text(t, teacher_tokenizer) for t in batch_out_texts]
            batch_in_texts = teacher_tokenizer.batch_decode(
                [ids for ids, _, _, _ in batch_inputs],
                skip_special_tokens=True,
            )
            batch_in_texts = [_strip_special_tokens_from_text(t, teacher_tokenizer) for t in batch_in_texts]

            lines = []
            for i in range(len(batch_inputs)):
                _, _, prompt_index, sample_index = batch_inputs[i]
                lines.append(json.dumps({
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "in_text": batch_in_texts[i],
                    "out_text": batch_out_texts[i],
                }) + "\n")
            with open(output_path, "a") as f:
                if HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.writelines(lines)
                    f.flush()
                finally:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        texts = _load_generations_jsonl(output_path, total_flat)
        if total_gen_chunk_num is not None and len(texts) != total_flat:
            print(f"Chunk {current_chunk_num}/{total_gen_chunk_num} done. Saved {len(texts)}/{total_flat} total. "
                  "Run all chunk jobs, then run without --total_gen_chunk_num to merge and train.")
            return Dataset.from_list([])
        metadata = {
            "teacher": teacher_name,
            "dataset": dataset_name,
            "split": split,
            "n_prompts": n_total,
            "n_texts": len(texts),
            "max_new_tokens": max_new_tokens,
            "gen_temperature": gen_temperature,
            "num_samples_per_prompt": num_samples_per_prompt,
            "seed": seed,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved {len(texts)} texts to {output_path}")

    if tokenized_dir.exists() and not force_regenerate:
        dataset = load_from_disk(str(tokenized_dir))
        print(f"Loaded tokenized dataset from {tokenized_dir}")
    else:
        dataset = _tokenize_texts_to_dataset(texts, student_tokenizer, student_max_length)
        dataset.save_to_disk(str(tokenized_dir))
        print(f"Saved tokenized dataset to {tokenized_dir}")

    return dataset


def prepare_dataset(
        dataset_name: str | None = None,
        max_samples: int | None = None,
        tokenizer: AutoTokenizer | None = None,
        max_length: int = 2048,
        force_prepare: bool = False,
        val_split: float | int = 0.1,
        test_split: float | int = 0.1,
        split_seed: int = 42,
        datadir: str | None = None,
) -> dict[str, Dataset]:
    """Prepare dataset for distillation.

    When the HuggingFace dataset has only one split, we split it into train/val/test.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for dataset preparation")
    if datadir is None:
        raise ValueError("datadir is required for dataset preparation")

    from pathlib import Path
    tokenizer_name = os.path.basename(tokenizer.name_or_path).replace("/", "_")
    split_key = f"v{val_split}_t{test_split}_s{split_seed}"
    cache_dir = Path(datadir) / "tokenized_maths_datasets" / f"{dataset_name}_{tokenizer_name}_{split_key}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_cache_path = cache_dir / "train"
    validation_cache_path = cache_dir / "validation"
    test_cache_path = cache_dir / "test"

    def _cache_exists(*paths):
        return all(p.exists() and any(p.iterdir()) for p in paths)

    print(f"Checking for cached tokenized dataset in {cache_dir}...")
    if not _cache_exists(train_cache_path, validation_cache_path, test_cache_path) or force_prepare:
        dataset = get_dataset(dataset_name, val_split, test_split, split_seed)
        print("Tokenizing dataset...")
        dataset["train"] = tokenize_dataset(dataset["train"], tokenizer, max_length=max_length)
        dataset["validation"] = tokenize_dataset(dataset["validation"], tokenizer, max_length=max_length)
        dataset["test"] = tokenize_dataset(dataset["test"], tokenizer, max_length=max_length)
        print(f"Saving tokenized dataset to cache: {cache_dir}")
        dataset["train"].save_to_disk(str(train_cache_path))
        dataset["validation"].save_to_disk(str(validation_cache_path))
        dataset["test"].save_to_disk(str(test_cache_path))
        print("Tokenized dataset cached successfully")

    print(f"Loading cached tokenized dataset from {cache_dir}...")
    try:
        dataset = {
            "train": load_from_disk(str(train_cache_path)),
            "validation": load_from_disk(str(validation_cache_path)),
            "test": load_from_disk(str(test_cache_path)),
        }
        print(f"Loaded cached tokenized dataset: train={len(dataset['train'])}, val={len(dataset['validation'])}, test={len(dataset['test'])}")
    except Exception as e:
        print(f"Warning: Failed to load cached dataset: {e}")
        print("Regenerating tokenized dataset...")
        dataset = get_dataset(dataset_name, val_split, test_split, split_seed)
        dataset["train"] = tokenize_dataset(dataset["train"], tokenizer, max_length=max_length)
        dataset["validation"] = tokenize_dataset(dataset["validation"], tokenizer, max_length=max_length)
        dataset["test"] = tokenize_dataset(dataset["test"], tokenizer, max_length=max_length)
        dataset["train"].save_to_disk(str(train_cache_path))
        dataset["validation"].save_to_disk(str(validation_cache_path))
        dataset["test"].save_to_disk(str(test_cache_path))

    if max_samples and max_samples > 0:
        for k in dataset.keys():
            dataset[k] = dataset[k].select(range(max_samples))

    return dataset
