import re
import json
from typing import Any
import numpy as np

import matplotlib.pyplot as plt

import torch

from transformers import AutoConfig, AutoModelForCausalLM
from tqdm import tqdm

from natual_pbat.constants import STUDENTS
from pyaromatics.hf_tools.utils import count_llm_parameters_noembs

model_eq = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "9.2306 * hidden_size**2 + 1926.8821 * hidden_size - 226840.5346",
    "moonshotai/Kimi-K2-Base": "9.4371 * hidden_size**2 + 2129.8179 * hidden_size - 331878.9518",
    "Goedel-LM/Goedel-Prover-V2-8B": "10.9561 * hidden_size**2 + 4881.6451 * hidden_size - 454229.1752",
    "Qwen/Qwen3-8B": "12.1166 * hidden_size**2 + 4035.1412 * hidden_size - 648284.6347",
}


def scaled_config(
        model_name: str, hidden_size: int, vocab_size: int = 32000, num_layers: int = 1, do_print=True,
) -> Any:
    # Load base config to get architecture info
    base_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if do_print:
        print('\n\nOriginal Config')
        print(base_config)

    # Get original config values
    original_hidden_size = getattr(base_config, 'hidden_size', 2048)
    original_num_layers = getattr(base_config, 'num_hidden_layers', getattr(base_config, 'num_layers', 32))
    original_num_heads = getattr(base_config, 'num_attention_heads', 32)
    original_intermediate_size = getattr(base_config, 'intermediate_size', None)
    original_proportion = original_intermediate_size / original_hidden_size

    original_kv_lora_rank = getattr(base_config, 'kv_lora_rank', 1)
    original_kv_proportion = original_kv_lora_rank / original_hidden_size

    original_moe_intermediate_size = getattr(base_config, 'moe_intermediate_size', 1)
    original_moe_proportion = original_moe_intermediate_size / original_hidden_size

    original_q_lora_rank = getattr(base_config, 'q_lora_rank', 1)

    # if not null divide it
    if original_q_lora_rank is None:
        original_q_proportion = None
    else:
        original_q_proportion = original_q_lora_rank / original_hidden_size

    # num_attention_heads largest 2 divisor that divides hidden_size
    num_attention_heads = max(
        h for h in range(1, hidden_size // 32)
        if hidden_size % h == 0
    )

    # Create new config by copying base config and modifying
    new_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    new_config.hidden_size = hidden_size

    new_config.num_hidden_layers = num_layers  # Just one layer for analysis
    new_config.num_layers = new_config.num_hidden_layers

    if hasattr(new_config, 'layer_types'):
        new_config.layer_types = [new_config.layer_types[0]] * num_layers

    new_config.num_attention_heads = num_attention_heads
    new_config.intermediate_size = int(original_proportion * hidden_size)
    new_config.vocab_size = vocab_size  # Keep vocab size constant
    new_config.max_position_embeddings = 40960
    new_config.kv_lora_rank = max(int(original_kv_proportion * hidden_size), 4)
    new_config.moe_intermediate_size = max(int(original_moe_proportion * hidden_size), 32)

    if original_q_proportion is None:
        new_config.q_lora_rank = None
    else:
        new_config.q_lora_rank = max(int(original_q_proportion * hidden_size), 4)

    # only qwen
    new_config.num_key_value_heads = max(num_attention_heads // 2, 1)

    if do_print:
        print('\n\nScaled Config')
        print(new_config)

    # del new_config.quantization_config

    return new_config


def analyze_layer_parameter_scaling(
        model_name: str,
        width_range: tuple[int, int] = (128, 1024),
        num_widths: int = 8,
        doplot: bool = False,
) -> str:
    """
    Analyze how parameters in one transformer layer scale with width (hidden_size).

    Args:
        model_name: HuggingFace model name
        width_range: Tuple of (min_width, max_width) to test
        num_widths: Number of different widths to test

    Returns:
        String describing the parameter scaling relationship
    """
    print(f"Analyzing parameter scaling for {model_name}...")

    # Generate test widths (logarithmically spaced for better coverage)
    min_width, max_width = width_range
    widths = np.logspace(
        np.log10(min_width),
        np.log10(max_width),
        num_widths,
        dtype=int
    ).tolist()
    widths = sorted(set(widths))  # Remove duplicates and sort

    print(f"Testing widths: {widths}")

    param_counts = []
    layer_configs = []

    for hidden_size in widths:

        # Create model from config
        config = scaled_config(model_name, hidden_size)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        if hidden_size == widths[0]:
            print(model)

        # remove embeddings and lm head to focus on transformer layer
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            del model.model.embed_tokens

        if hasattr(model, 'lm_head'):
            del model.lm_head

        if hidden_size == widths[0]:
            print(model)

        # Get the first transformer layer
        # Common patterns: model.layers[0], model.model.layers[0], etc.
        layer = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer = model.model.layers[0]
        elif hasattr(model, 'layers'):
            layer = model.layers[0]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layer = model.transformer.h[0]

        if layer is None:
            print(f"Warning: Could not find transformer layer for width {hidden_size}")
            continue

        # Count parameters in this layer
        layer_params, _ = count_llm_parameters_noembs(layer)
        param_counts.append(layer_params)
        layer_configs.append({
            'hidden_size': hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'intermediate_size': config.intermediate_size,
        })

        print(f"  Width {hidden_size}: {layer_params} parameters")

        # Clean up model to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if len(param_counts) < 3:
        return f"Error: Need at least 3 data points, got {len(param_counts)}"

    # Fit polynomial to understand scaling
    # Try quadratic fit first (most common: O(hidden_size^2))
    widths_array = np.array([c['hidden_size'] for c in layer_configs])
    params_array = np.array(param_counts)

    # Fit polynomial: params = a * width^2 + b * width + c
    coeffs = np.polyfit(widths_array, params_array, deg=2)
    a, b, c = coeffs

    # Check if linear or quadratic dominates
    # Evaluate at max width to see which term is larger
    max_width = widths_array.max()
    quadratic_term = a * max_width ** 2
    linear_term = b * max_width
    # constant_term = c

    if doplot:
        plt.figure(figsize=(8, 6))
        plt.scatter(widths_array, params_array, label='Measured Parameters', color='blue')

        fitted_params = a * widths_array ** 2 + b * widths_array + c
        plt.plot(widths_array, fitted_params, label='Fitted Quadratic', color='red')

        plt.xlabel('Hidden Size (width)')
        plt.ylabel('Number of Parameters in Layer')
        plt.title(f'Parameter Scaling for {model_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Build formula string
    terms = []
    if abs(a) > 1e-6:
        if abs(a - round(a)) < 0.01:
            a_str = f"{int(round(a))}"
        else:
            a_str = f"{a:.4f}"
        terms.append(f"{a_str} * hidden_size^2")

    if abs(b) > 1e-6:
        if abs(b - round(b)) < 0.01:
            b_str = f"{int(round(b))}"
        else:
            b_str = f"{b:.4f}"
        terms.append(f"{b_str} * hidden_size")

    if abs(c) > 1e-6:
        if abs(c - round(c)) < 0.01:
            c_str = f"{int(round(c))}"
        else:
            c_str = f"{c:.4f}"
        terms.append(c_str)

    formula = " + ".join(terms) if terms else "0"

    # Determine dominant scaling
    if abs(quadratic_term) > abs(linear_term) * 10:
        scaling_type = "quadratic (O(hidden_size²))"
    elif abs(linear_term) > abs(quadratic_term) * 10:
        scaling_type = "linear (O(hidden_size))"
    else:
        scaling_type = "mixed (quadratic + linear)"

    # Create result string
    result = (
        f"Parameter scaling for {model_name}:\n"
        f"  Formula: params = {formula}\n"
        f"  Scaling: {scaling_type}\n"
        f"  Tested widths: {widths_array.tolist()}\n"
        f"  Parameter counts: {params_array.tolist()}\n"
        f"  Config details:\n"
    )

    for i, config in enumerate(layer_configs):
        result += (
            f"    hidden_size={config['hidden_size']}: "
            f"heads={config['num_attention_heads']}, "
            f"intermediate={config['intermediate_size']} => "
            f"{param_counts[i]:,} params\n"
        )

    print(result)
    return result


def evaluate_equation(eq_str, hidden_sizes):
    results = []
    for h in hidden_sizes:
        # define 'hidden_size' in the local scope for eval
        hidden_size = h
        # safely evaluate the expression
        value = eval(eq_str, {"__builtins__": None}, {"hidden_size": hidden_size})
        results.append(value)
    return results


def extrapolations(num_widths=10, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    r = (1024 / 128) ** (1 / 7)  # adjust ratio to fit desired range
    print('r=', r)
    widths = [128]
    for _ in range(num_widths):
        new_width = widths[-1] * r
        widths.append(new_width)

    widths = [int(round(w)) for w in widths]
    print(widths)

    eq = model_eq[model_name]
    params = evaluate_equation(eq, widths)

    for h, p in zip(widths, params):
        print(f"hidden_size={h:>4} → params={p:,.2f}")


def find_small_config(model_name="Qwen/Qwen3-8B", target_params=100_000, error_percent=20, attempts=10):
    initial_width = 128

    too_large_widths = []
    too_small_widths = []
    hidden_size = initial_width
    for _attempt in range(attempts):
        print(f"\nAttempt {_attempt + 1}:")

        # Create model from config
        config = scaled_config(model_name, hidden_size, num_layers=1, do_print=False)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        # remove embeddings and lm head to focus on transformer layer
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            del model.model.embed_tokens

        if hasattr(model, 'lm_head'):
            del model.lm_head

        # Get the first transformer layer
        # Common patterns: model.layers[0], model.model.layers[0], etc.
        layer = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer = model.model.layers[0]
        elif hasattr(model, 'layers'):
            layer = model.layers[0]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layer = model.transformer.h[0]

        # Count parameters in this layer
        layer_params, _ = count_llm_parameters_noembs(layer)


        print(f"  Width {hidden_size}: {layer_params} parameters")
        if target_params<= layer_params <= target_params *(1 + error_percent/100):
            print(f"Found suitable config with hidden_size={hidden_size} yielding {layer_params} parameters.")
            print(f"   after {_attempt + 1} attempts.")
            break

        elif target_params > layer_params:
            too_small_widths.append(hidden_size)
            if too_large_widths:
                hidden_size = (hidden_size + min(too_large_widths)) // 2
            else:
                hidden_size *= 2

        else:
            too_large_widths.append(hidden_size)
            if too_small_widths:
                hidden_size = (hidden_size + max(too_small_widths)) // 2
            else:
                hidden_size //= 2

        # Clean up model to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


    # Clean up model to free memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return config


if __name__ == "__main__":
    pass

    # "max_position_embeddings": 40960,
    # analyze_layer_parameter_scaling("Goedel-LM/Goedel-Prover-V2-8B")
    # Parameter scaling for Goedel-LM/Goedel-Prover-V2-8B:
    #   Formula: params = 10.9561 * hidden_size^2 + 4881.6451 * hidden_size + -454229.1752
    #   Scaling: mixed (quadratic + linear)
    #   Tested widths: [127, 172, 231, 312, 420, 565, 760, 1024]
    #   Parameter counts: [438279, 795240, 1131463, 2154928, 3739096, 4754731, 10647856, 15730944]
    #   Config details:
    #     hidden_size=127: heads=1, intermediate=381 => 438,279 params
    #     hidden_size=172: heads=4, intermediate=516 => 795,240 params
    #     hidden_size=231: heads=3, intermediate=693 => 1,131,463 params
    #     hidden_size=312: heads=8, intermediate=936 => 2,154,928 params
    #     hidden_size=420: heads=12, intermediate=1260 => 3,739,096 params
    #     hidden_size=565: heads=5, intermediate=1695 => 4,754,731 params
    #     hidden_size=760: heads=20, intermediate=2280 => 10,647,856 params
    #     hidden_size=1024: heads=16, intermediate=3072 => 15,730,944 params

    # "max_position_embeddings": 40960,
    # analyze_layer_parameter_scaling("moonshotai/Kimi-Linear-48B-A3B-Instruct")
    # Parameter scaling for moonshotai/Kimi-K2-Base:
    #   Formula: params = 9.4371 * hidden_size^2 + 2129.8179 * hidden_size + -331878.9518
    #   Scaling: mixed (quadratic + linear)
    #   Tested widths: [127, 172, 231, 312, 420, 565, 760, 1024]
    #   Parameter counts: [160940, 375728, 571184, 1264728, 2383680, 3118516, 7513816, 11524388]
    #   Config details:
    #     hidden_size=127: heads=1, intermediate=326 => 160,940 params
    #     hidden_size=172: heads=4, intermediate=442 => 375,728 params
    #     hidden_size=231: heads=3, intermediate=594 => 571,184 params
    #     hidden_size=312: heads=8, intermediate=802 => 1,264,728 params
    #     hidden_size=420: heads=12, intermediate=1080 => 2,383,680 params
    #     hidden_size=565: heads=5, intermediate=1452 => 3,118,516 params
    #     hidden_size=760: heads=20, intermediate=1954 => 7,513,816 params
    #     hidden_size=1024: heads=16, intermediate=2633 => 11,524,388 params

    # "max_position_embeddings": 40960,
    # analyze_layer_parameter_scaling("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # Parameter scaling for TinyLlama/TinyLlama-1.1B-Chat-v1.0:
    #   Formula: params = 9.2306 * hidden_size^2 + 1926.8821 * hidden_size + -226840.5346
    #   Scaling: mixed (quadratic + linear)
    #   Tested widths: [127, 172, 231, 312, 420, 565, 760, 1024]
    #   Parameter counts: [214503, 420540, 647493, 1282944, 2316300, 3284345, 7101440, 11274240]
    #   Config details:
    #     hidden_size=127: heads=1, intermediate=349 => 214,503 params
    #     hidden_size=172: heads=4, intermediate=473 => 420,540 params
    #     hidden_size=231: heads=3, intermediate=635 => 647,493 params
    #     hidden_size=312: heads=8, intermediate=858 => 1,282,944 params
    #     hidden_size=420: heads=12, intermediate=1155 => 2,316,300 params
    #     hidden_size=565: heads=5, intermediate=1553 => 3,284,345 params
    #     hidden_size=760: heads=20, intermediate=2090 => 7,101,440 params
    #     hidden_size=1024: heads=16, intermediate=2816 => 11,274,240 params

    #     30 layers for 1B -> 34 million parameters per layer
    # extrapolations(num_widths=10, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0") # 35110251
    # hidden_size=1855 → params=35,110,251.13 - n_layers=29
    # extrapolations(num_widths=10, model_name="moonshotai/Kimi-K2-Base") # 36092230
    # hidden_size=1855 → params=36,092,230.28 - n_layers=28
    # extrapolations(num_widths=10, model_name="Goedel-LM/Goedel-Prover-V2-8B") # 46301436
    # hidden_size=1855 → params=46,301,436.49 - n_layers=22

    # analyze_layer_parameter_scaling("meta-llama/Llama-3.2-1B")
    # Parameter scaling for meta-llama/Llama-3.2-1B:
    #   Formula: params = 13.5583 * hidden_size^2 + 2018.5706 * hidden_size + -324270.3174
    #   Scaling: mixed (quadratic + linear)
    #   Tested widths: [127, 172, 231, 312, 420, 565, 760, 1024]
    #   Parameter counts: [226314, 487448, 759066, 1647984, 3085320, 4338070, 9851120, 15730688]
    #   Config details:
    #     hidden_size=127: heads=1, intermediate=508 => 226,314 params
    #     hidden_size=172: heads=4, intermediate=688 => 487,448 params
    #     hidden_size=231: heads=3, intermediate=924 => 759,066 params
    #     hidden_size=312: heads=8, intermediate=1248 => 1,647,984 params
    #     hidden_size=420: heads=12, intermediate=1680 => 3,085,320 params
    #     hidden_size=565: heads=5, intermediate=2260 => 4,338,070 params
    #     hidden_size=760: heads=20, intermediate=3040 => 9,851,120 params
    #     hidden_size=1024: heads=16, intermediate=4096 => 15,730,688 params

    # analyze_layer_parameter_scaling("Qwen/Qwen3-8B")

    # Parameter scaling for Qwen/Qwen3-8B:
    #   Formula: params = 12.1166 * hidden_size^2 + 4035.1412 * hidden_size + -648284.6347
    #   Scaling: mixed (quadratic + linear)
    #   Tested widths: [127, 172, 231, 312, 420, 565, 760, 1024]
    #   Parameter counts: [210695, 531048, 717511, 1835440, 3524056, 3886891, 11036976, 15730944]
    #   Config details:
    #     hidden_size=127: heads=1, intermediate=381 => 210,695 params
    #     hidden_size=172: heads=4, intermediate=516 => 531,048 params
    #     hidden_size=231: heads=3, intermediate=693 => 717,511 params
    #     hidden_size=312: heads=8, intermediate=936 => 1,835,440 params
    #     hidden_size=420: heads=12, intermediate=1260 => 3,524,056 params
    #     hidden_size=565: heads=5, intermediate=1695 => 3,886,891 params
    #     hidden_size=760: heads=20, intermediate=2280 => 11,036,976 params
    #     hidden_size=1024: heads=16, intermediate=3072 => 15,730,944 params
    # extrapolations(num_widths=10, model_name="Qwen/Qwen3-8B")  # 36092230
    find_small_config()