"""Evaluation helpers for distillation."""

import gc, time, os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from pyaromatics.hf_tools.distillation.trainer import DistillationTrainer


def do_evaluation(
        teacher_model: AutoModelForCausalLM | None,
        student_model: AutoModelForCausalLM,
        validation_dataset,
        tokenizer: AutoTokenizer,
        training_max_length: int | None = None,
        use_fp16: bool = False,
        use_bf16: bool = False,
        reduced_validation: bool = False,
        expdir: str | None = None,
) -> dict[str, float]:
    """Evaluate teacher (if provided) and student on the validation dataset."""
    if expdir is None:
        raise ValueError("expdir must be provided for evaluation output")

    if reduced_validation:
        reduced_size = min(7000, len(validation_dataset))
        print(f"Using reduced validation dataset of size {reduced_size} for evaluation")
        validation_dataset = validation_dataset.select(range(reduced_size))

    eval_args = TrainingArguments(
        output_dir=expdir,
        per_device_eval_batch_size=1,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
    )

    teacher_validation_loss = float("inf")
    if teacher_model is not None:
        teacher_trainer_kwargs = {
            "model": teacher_model,
            "args": eval_args,
            "eval_dataset": validation_dataset,
            "tokenizer": tokenizer,
            "training_max_length": training_max_length,
        }
        print("Evaluating teacher model on validation dataset...")
        teacher_validator = DistillationTrainer(**teacher_trainer_kwargs)
        teacher_eval_output = teacher_validator.evaluate()
        teacher_validation_loss = teacher_eval_output.get("eval_loss", float("inf"))

    print("Evaluating student model on validation dataset...")
    student_trainer_kwargs = {
        "model": student_model,
        "args": eval_args,
        "eval_dataset": validation_dataset,
        "tokenizer": tokenizer,
        "training_max_length": training_max_length,
    }
    student_validator = DistillationTrainer(**student_trainer_kwargs)
    student_eval_output = student_validator.evaluate()
    student_validation_loss = student_eval_output.get("eval_loss", float("inf"))

    results: dict[str, float] = {"student_validation_loss": student_validation_loss}
    if teacher_model is not None:
        results["teacher_validation_loss"] = teacher_validation_loss

    print(f"\nEvaluation Results:")
    if teacher_model is not None:
        print(f"  Teacher validation loss: {teacher_validation_loss:.4f}")
    print(f"  Student validation loss: {student_validation_loss:.4f}")

    return results
