"""Helper functions for model distillation."""

from __future__ import annotations

import socket, time, gc
from typing import Any
import torch.nn as nn

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def is_thalia() -> bool:
    """Check if running on thalia host."""
    return socket.gethostname() == "thalia"


def get_config_dtype_for_creation(config: Any) -> torch.dtype | None:
    """Extract dtype from config for model creation."""
    if not hasattr(config, 'dtype'):
        return None
    config_dtype = config.dtype
    if isinstance(config_dtype, str):
        if config_dtype == "bfloat16":
            return torch.bfloat16
        elif config_dtype == "float16":
            return torch.float16
    elif config_dtype == torch.bfloat16:
        return torch.bfloat16
    elif config_dtype == torch.float16:
        return torch.float16
    return None


def create_model_with_dtype(
        config: Any,
        is_thalia_host: bool,
        trust_remote_code: bool = True,
) -> AutoModelForCausalLM:
    """Create model from config, respecting dtype on thalia."""
    if is_thalia_host:
        config_dtype = get_config_dtype_for_creation(config)
        if config_dtype is not None:
            return AutoModelForCausalLM.from_config(
                config, dtype=config_dtype, trust_remote_code=trust_remote_code
            )
    return AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)


def determine_load_dtype(
        student_path: Any,
        use_fp16: bool,
        is_thalia_host: bool,
) -> torch.dtype:
    """Determine dtype to use when loading a model."""
    if not is_thalia_host:
        return torch.float16 if use_fp16 else torch.float32

    student_config = AutoConfig.from_pretrained(student_path, trust_remote_code=True)
    config_dtype = getattr(student_config, 'dtype', None)
    print(f"Config dtype attribute: {config_dtype} (type: {type(config_dtype)})")

    is_bfloat16_config = False
    if config_dtype is not None:
        if isinstance(config_dtype, str) and config_dtype == "bfloat16":
            is_bfloat16_config = True
        elif config_dtype == torch.bfloat16:
            is_bfloat16_config = True
        elif str(config_dtype) == "bfloat16":
            is_bfloat16_config = True

    if is_bfloat16_config and use_fp16:
        print(f"Config specifies bfloat16, loading model with bfloat16 dtype")
        return torch.bfloat16
    elif use_fp16:
        print(f"Config does not specify bfloat16, loading model with float16 dtype")
        return torch.float16
    else:
        print(f"Mixed precision disabled, loading model with float32 dtype")
        return torch.float32


def enhance_logs_with_custom_metrics(
        logs: dict[str, float],
        current_distillation_loss: float | None,
        current_task_loss: float | None,
        trainer_instance: Any,
) -> dict[str, float]:
    """Enhance logs dictionary with custom metrics for distillation."""
    if current_distillation_loss is not None:
        logs["distillation_loss"] = current_distillation_loss
    if current_task_loss is not None:
        logs["task_loss"] = current_task_loss

    if "learning_rate" not in logs:
        if hasattr(trainer_instance, "lr_scheduler") and trainer_instance.lr_scheduler is not None:
            lr = trainer_instance.lr_scheduler.get_last_lr()
            if lr:
                logs["learning_rate"] = lr[0] if isinstance(lr, list) else lr
        elif hasattr(trainer_instance.args, "learning_rate"):
            logs["learning_rate"] = trainer_instance.args.learning_rate

    if "epoch" not in logs:
        logs["epoch"] = trainer_instance.state.epoch if hasattr(trainer_instance.state, "epoch") else 0.0

    logs.pop("global_step", None)
    return logs


def determine_training_precision(
        student_model: AutoModelForCausalLM,
        use_fp16: bool,
        is_thalia_host: bool,
) -> tuple[bool, bool]:
    """Determine which mixed precision to use for training (bf16 or fp16)."""
    if not is_thalia_host:
        return False, use_fp16

    config_dtype_str = getattr(student_model.config, 'dtype', None)
    student_dtype = next(student_model.parameters()).dtype

    config_dtype = None
    if config_dtype_str:
        if isinstance(config_dtype_str, str):
            if config_dtype_str == "bfloat16":
                config_dtype = torch.bfloat16
            elif config_dtype_str == "float16":
                config_dtype = torch.float16
            elif config_dtype_str == "float32":
                config_dtype = torch.float32

    effective_dtype = config_dtype if (config_dtype == torch.bfloat16) else student_dtype

    use_bf16 = False
    use_fp16_actual = False

    if use_fp16:
        if effective_dtype == torch.bfloat16 or student_dtype == torch.bfloat16:
            use_bf16 = True
            print(f"Detected bfloat16 model dtype (config: {config_dtype_str}, params: {student_dtype}), using bf16 mixed precision")
        elif effective_dtype == torch.float16 or student_dtype == torch.float16:
            use_fp16_actual = True
            print(f"Detected float16 model dtype (config: {config_dtype_str}, params: {student_dtype}), using fp16 mixed precision")
        else:
            if torch.cuda.is_bf16_supported():
                use_bf16 = True
                print(f"Model is float32, using bf16 mixed precision (GPU supports bf16)")
            else:
                use_fp16_actual = True
                print(f"Model is float32, using fp16 mixed precision (GPU doesn't support bf16)")

    return use_bf16, use_fp16_actual


def count_llm_parameters_noembs(model: nn.Module):
    embedding_param_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                embedding_param_ids.add(id(p))

    total_params = 0
    total_params_noembs = 0
    for p in model.parameters():
        n = p.numel()
        total_params += n
        if id(p) not in embedding_param_ids:
            total_params_noembs += n
    return total_params_noembs, total_params


def count_llm_parameters_detailed(model: nn.Module):
    """Count parameters with detailed breakdown: total, no_embs, no_embs_output."""
    embedding_param_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                embedding_param_ids.add(id(p))

    output_param_ids = set()
    output_layer_names = ['lm_head', 'embed_out', 'output_projection', 'head']
    for name, module in model.named_modules():
        if any(layer_name in name.lower() for layer_name in output_layer_names):
            if isinstance(module, nn.Linear):
                for p in module.parameters(recurse=False):
                    output_param_ids.add(id(p))

    total_params = 0
    total_params_noembs = 0
    total_params_noembs_output = 0
    for p in model.parameters():
        n = p.numel()
        total_params += n
        if id(p) not in embedding_param_ids:
            total_params_noembs += n
            if id(p) not in output_param_ids:
                total_params_noembs_output += n

    return {
        'total': total_params,
        'no_embs': total_params_noembs,
        'no_embs_output': total_params_noembs_output,
    }


def generate_text(model, tokenizer, dataset_sample, top_p=1.0, top_k=50, temperature=1.0, max_length=50, num_beams=1):
    device = next(model.parameters()).device
    input_ids = torch.tensor(dataset_sample['input_ids']).to(device)
    attention_mask = torch.tensor(dataset_sample['attention_mask']).to(device)
    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        do_sample=True,
        no_repeat_ngram_size=2,
    )
    return (
        tokenizer.decode(input_ids[0], skip_special_tokens=True),
        tokenizer.decode(generated_ids[0], skip_special_tokens=True),
    )
