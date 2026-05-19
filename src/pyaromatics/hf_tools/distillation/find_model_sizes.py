import re
import json
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from tqdm import tqdm

from pyaromatics.hf_tools.distillation.helpers import count_llm_parameters_noembs

model_eq = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "9.2306 * hidden_size**2 + 1926.8821 * hidden_size - 226840.5346",
    "moonshotai/Kimi-K2-Base": "9.4371 * hidden_size**2 + 2129.8179 * hidden_size - 331878.9518",
    "Goedel-LM/Goedel-Prover-V2-8B": "10.9561 * hidden_size**2 + 4881.6451 * hidden_size - 454229.1752",
    "Qwen/Qwen3-8B": "12.1166 * hidden_size**2 + 4035.1412 * hidden_size - 648284.6347",
}


def scaled_config(
        model_name: str, hidden_size: int, vocab_size: int = 32000, num_layers: int = 1, do_print=True,
) -> Any:
    base_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if do_print:
        print('\n\nOriginal Config')
        print(base_config)

    original_hidden_size = getattr(base_config, 'hidden_size', 2048)
    original_num_layers = getattr(base_config, 'num_hidden_layers', getattr(base_config, 'num_layers', 32))
    original_intermediate_size = getattr(base_config, 'intermediate_size', None)
    original_proportion = original_intermediate_size / original_hidden_size

    original_kv_lora_rank = getattr(base_config, 'kv_lora_rank', 1)
    original_kv_proportion = original_kv_lora_rank / original_hidden_size

    original_moe_intermediate_size = getattr(base_config, 'moe_intermediate_size', 1)
    original_moe_proportion = original_moe_intermediate_size / original_hidden_size

    original_q_lora_rank = getattr(base_config, 'q_lora_rank', 1)
    original_q_proportion = (original_q_lora_rank / original_hidden_size) if original_q_lora_rank is not None else None

    num_attention_heads = max(
        h for h in range(1, hidden_size // 32) if hidden_size % h == 0
    )

    new_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    new_config.hidden_size = hidden_size
    new_config.num_hidden_layers = num_layers
    new_config.num_layers = new_config.num_hidden_layers
    if hasattr(new_config, 'layer_types'):
        new_config.layer_types = [new_config.layer_types[0]] * num_layers
    new_config.num_attention_heads = num_attention_heads
    new_config.num_heads = num_attention_heads
    new_config.intermediate_size = int(original_proportion * hidden_size)
    new_config.vocab_size = vocab_size
    new_config.max_position_embeddings = 40960
    new_config.kv_lora_rank = max(int(original_kv_proportion * hidden_size), 4)
    new_config.moe_intermediate_size = max(int(original_moe_proportion * hidden_size), 32)
    if original_q_proportion is None:
        new_config.q_lora_rank = None
    else:
        new_config.q_lora_rank = max(int(original_q_proportion * hidden_size), 4)
    new_config.num_key_value_heads = max(num_attention_heads // 2, 1)

    if hasattr(new_config, 'expand'):
        new_config.head_dim = new_config.hidden_size * new_config.expand // new_config.num_heads
        new_config.n_groups = new_config.num_heads // 2

    if do_print:
        print('\n\nScaled Config')
        print(new_config)

    return new_config
