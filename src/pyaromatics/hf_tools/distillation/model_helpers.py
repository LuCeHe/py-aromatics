"""Model loading helpers for distillation: teacher + student model creation."""

import os, shutil, socket
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from pyaromatics.hf_tools.distillation.find_model_sizes import scaled_config
from pyaromatics.hf_tools.distillation.helpers import (
    is_thalia,
    create_model_with_dtype,
    determine_load_dtype,
    count_llm_parameters_noembs,
)
from pyaromatics.hf_tools.utils import get_pretrained_model


def get_model_max_length(model: AutoModelForCausalLM) -> int:
    """Get the maximum sequence length supported by a model."""
    config = model.config
    max_length = getattr(config, 'max_position_embeddings', None)
    if max_length is None:
        max_length = getattr(config, 'max_seq_length', None)
    if max_length is None:
        max_length = getattr(config, 'n_positions', None)
    if max_length is None:
        raise ValueError("Could not determine model's maximum sequence length")
    return max_length


def get_models(
        student_key: str,
        student_size: str,
        teacher_model_name: str,
        seed: int,
        max_length: int,
        use_fp16: bool,
        use_bf16: bool,
        is_thalia_host: bool,
        students_map: dict[str, str],
        students_pretrained_map: dict[str, str],
        arch_hypers_map: dict[str, dict[str, dict[str, int]]],
        datadir: str,
        distilled_dir: str,
) -> tuple[AutoModelForCausalLM, AutoModelForCausalLM, dict[str, Any]]:
    """Load teacher and student models for distillation.

    Args:
        student_key: Key in students_map or students_pretrained_map.
        student_size: Size label (e.g. '1M', '10M').
        teacher_model_name: HuggingFace model ID for the teacher.
        seed: Random seed.
        max_length: Maximum sequence length.
        use_fp16, use_bf16: Precision flags.
        is_thalia_host: Whether running on thalia.
        students_map: Maps student key -> HF model ID (arch template).
        students_pretrained_map: Maps student key -> HF model ID (pretrained).
        arch_hypers_map: Maps student key -> size -> {hidden_size, num_layers}.
        datadir: Data directory for caching.
        distilled_dir: Directory for storing distilled models.

    Returns:
        (teacher_model, student_model, results_dict)
    """
    teacher_model = get_pretrained_model(
        teacher_model_name,
        save_dir=datadir,
        dtype=torch.float16 if use_fp16 or use_bf16 else torch.float32,
        device_map="auto",
    )

    is_pretrained_student = student_key in students_pretrained_map

    if is_pretrained_student:
        student_model_name = students_pretrained_map[student_key]
        print(f"Loading pretrained student: {student_model_name} (native tokenizer)")
        from transformers import AutoTokenizer
        load_dtype = torch.float16 if use_fp16 or use_bf16 else torch.float32
        student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=load_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        student_tokenizer = AutoTokenizer.from_pretrained(
            student_model_name,
            trust_remote_code=True,
        )
        if student_tokenizer.pad_token is None:
            student_tokenizer.pad_token = student_tokenizer.eos_token
        student_model.get_input_embeddings().requires_grad_(True)
        student_model.get_output_embeddings().requires_grad_(True)
    else:
        student_tokenizer = None
        student_model_name = students_map[student_key]
        student_path = Path(distilled_dir) / "random_students" / f"{student_key}_{student_size}_seed{seed}"
        if student_path.exists():
            print(f"Removing existing student model directory: {student_path}")
            shutil.rmtree(student_path)
        student_path.mkdir(parents=True, exist_ok=True)

        if not (student_path / "config.json").exists():
            print(f"Loading student model: {student_model_name}")
            hidden_size = arch_hypers_map[student_key][student_size]['hidden_size']
            num_layers = arch_hypers_map[student_key][student_size]['num_layers']
            config = scaled_config(
                model_name=student_model_name,
                hidden_size=hidden_size,
                vocab_size=teacher_model.config.vocab_size,
                num_layers=num_layers,
            )
            model = create_model_with_dtype(config, is_thalia_host, trust_remote_code=True)
            model.save_pretrained(
                student_path,
                safe_serialization=True,
                copy_dynamic_modules=True,
            )

        load_dtype = determine_load_dtype(student_path, use_fp16, is_thalia_host)
        student_model = AutoModelForCausalLM.from_pretrained(
            student_path,
            dtype=load_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        student_model.get_input_embeddings().requires_grad_(True)
        student_model.get_output_embeddings().requires_grad_(True)

    print(f"\nTeacher model: {teacher_model_name}")
    print(f"Student model: {student_model_name}" + (f" (size: {student_size})" if not is_pretrained_student else " (pretrained)"))

    results = {}
    teacher_params_noembs, teacher_params = count_llm_parameters_noembs(teacher_model)
    student_params_noembs, student_params = count_llm_parameters_noembs(student_model)
    print(f"Teacher parameters (no embs): {teacher_params_noembs / 1e6:.2f}M, (total: {teacher_params / 1e6:.2f}M)")
    print(f"Student parameters (no embs): {student_params_noembs / 1e6:.2f}M, (total: {student_params / 1e6:.2f}M)")
    results["teacher_params"] = teacher_params
    results["student_params"] = student_params
    results["teacher_params_noembs"] = teacher_params_noembs
    results["student_params_noembs"] = student_params_noembs

    teacher_max_length = get_model_max_length(teacher_model)
    student_max_length = get_model_max_length(student_model)

    if is_pretrained_student:
        tokenization_max_length = max(teacher_max_length, max_length)
        training_max_length = min(student_max_length, max_length)
        results["is_pretrained_student"] = True
        results["student_tokenizer"] = student_tokenizer
    else:
        tokenization_max_length = max(teacher_max_length, student_max_length, max_length)
        if 'DESKTOP' in socket.gethostname():
            tokenization_max_length = min(teacher_max_length, student_max_length, max_length)
        training_max_length = min(teacher_max_length, student_max_length, max_length)

    print(f"Model max lengths - Teacher: {teacher_max_length}, Student: {student_max_length}")
    print(f"Tokenization max length: {tokenization_max_length}")
    print(f"Training max length: {training_max_length}")

    results["tokenization_max_length"] = tokenization_max_length
    results["training_max_length"] = training_max_length

    return teacher_model, student_model, results
