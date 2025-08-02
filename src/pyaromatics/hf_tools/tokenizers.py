
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def get_tokenizer(model_id, notes, save_dir=None):
    if save_dir is None:
        raise ValueError("save_dir must be specified")

    tokenizer_id = model_id
    if 'mamba2' in model_id or 'ourllm' in model_id:
        tokenizer_id = "EleutherAI/gpt-neox-20b"

    if 'Qwen3' in model_id:
        tokenizer_id = "Qwen/Qwen2-0.5b"

    if 'byt5' in notes:
        tokenizer_id = "google/byt5-small"

    tokenizer_path = os.path.join(save_dir, tokenizer_id.replace('/', '-') + '-tokenizer')
    tokenizer_args = {
        # 'padding': 'max_length', 'max_length': 175,
        # 'pad_to_multiple_of':8,
        'truncation_side': 'left'
    }
    if not os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True, **tokenizer_args,
        )
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_args)

    # tokenizer details
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation = True

    # print('here?')
    # tokenizer.padding = 'max_length'
    # tokenizer.max_length = 175
    # tokenizer.pad_to_multiple_of = 8
    # tokenizer.pad_to_max_length = True
    # tokenizer.model_max_length = 175
    return tokenizer

def get_pretrained_model(model_id='gpt2', save_dir=None):
    if save_dir is None:
        raise ValueError("save_dir must be specified")
    model_path = os.path.join(save_dir, model_id.replace('/', '-') + '-model')

    if not os.path.exists(model_path):
        print('Downloading Model')
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, offload_buffers=True,
        )
        model.save_pretrained(model_path)
    else:
        print('Loading Model')
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16, offload_buffers=True,
        )

    return model
