import os, socket, base64, tempfile, re
import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM


def connected_to_internet():
    try:
        # Attempt to connect to a well-known server (Google's DNS server)
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False


def get_hf_key(savekey_dir):
    path_key = os.path.join(savekey_dir, 'hfstuff.txt')
    if not os.path.exists(path_key):
        token = input(f"Enter your HF token: ")
        encoded = base64.b64encode(token.encode('utf-8')).decode('utf-8')

        with open(path_key, 'w') as f:
            f.write(encoded)

    with open(path_key, 'r') as f:
        encoded = f.read()
    token = base64.b64decode(encoded).decode('utf-8')
    print('Using HF token from', token)

    from huggingface_hub import login

    # detect internet connection
    if connected_to_internet():
        print('Logging in to HF...')
        login(token=token)

    # set the environment variable
    os.environ['HF_AUTH_TOKEN'] = token
    os.system(f"export HF_AUTH_TOKEN={token}")
    return token


def get_tokenizer(model_id, notes='', save_dir=None, max_seq_length=None):
    if save_dir is None:
        raise ValueError("save_dir must be specified")

    from transformers import AutoTokenizer

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
        'truncation_side': 'left',
        'trust_remote_code': True,
    }
    if not os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, **tokenizer_args,
        )
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_args)

    # tokenizer details
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation = True

    if not max_seq_length is None:
        tokenizer.model_max_length = max_seq_length
        tokenizer.max_length = max_seq_length

    return tokenizer


def ensure_model_local(model_id, model_path):
    from huggingface_hub import snapshot_download

    if os.path.exists(model_path):
        print("Model already cached:", model_path)
        return model_path

    print("Downloading model without loading into memory:", model_id)
    cache_dir = snapshot_download(
        repo_id=model_id,
        local_dir=model_path,
        local_dir_use_symlinks=False  # copy instead of symlinking
    )
    print("Saved model to:", cache_dir)
    return cache_dir


class PatchedAutoModelForCausalLM(AutoModelForCausalLM):
    def forward(self, *args, **kwargs):
        kwargs.pop("num_items_in_batch", None)  # safely ignore it
        return super().forward(*args, **kwargs)


class PatchedAutoModelForMaskedLM(AutoModelForMaskedLM):
    def forward(self, *args, **kwargs):
        kwargs.pop("num_items_in_batch", None)  # safely ignore it
        return super().forward(*args, **kwargs)


def get_pretrained_model(
        model_id='gpt2', save_dir=None, return_path=False, offload_dir=None,
        dtype=torch.float32,
        device_map='auto',

):
    if offload_dir is None:
        # tmp dir in save_dir/offloads
        offload_dir = os.path.join(save_dir, 'offloads')
        os.makedirs(offload_dir, exist_ok=True)

    if save_dir is None:
        raise ValueError("save_dir must be specified")

    model_path = os.path.join(save_dir, model_id.replace('/', '-') + '-model')

    if return_path and os.path.exists(model_path):
        return model_path

    model_path = ensure_model_local(model_id, model_path)
    kwargs = {'device_map': device_map, 'dtype': dtype}
    if not offload_dir is None:
        kwargs = {
            'device_map': 'auto',
            'offload_buffers': True,
            'offload_state_dict': True,
            'offload_folder': offload_dir,
        }

    if 'gemma' in model_id.lower():
        kwargs['attn_implementation'] = 'eager'

    if model_id == 'openbmb/MiniCPM-2B-sft-bf16':
        from pyaromatics.hf_tools.models.modeling_minicpm import MiniCPMForCausalLM
        AutoM = lambda x, trust_remote=False: MiniCPMForCausalLM.from_pretrained(
            x, trust_remote_code=trust_remote, **kwargs
        )

    elif 'bert' in model_id.lower():
        AutoM = lambda x, trust_remote=False: AutoModelForMaskedLM.from_pretrained(
            x, output_hidden_states=True, trust_remote_code=trust_remote, **kwargs
        )

    else:
        AutoM = lambda x, trust_remote=False: AutoModelForCausalLM.from_pretrained(
            x, trust_remote_code=trust_remote, **kwargs
        )

    trust_remote = True
    if os.path.exists(model_path):
        print('Loading Model -', model_id)
        model_id = model_path
        trust_remote = True
    else:
        print('Downloading Model -', model_id)

    model = AutoM(model_id, trust_remote=trust_remote)

    if not os.path.exists(model_path):
        print('Saving Model -', model_path)
        model.save_pretrained(model_path)

    return model


def count_llm_parameters_noembs(model: AutoModelForCausalLM, do_print=False) -> (int, int):
    # remove embedding and lm head parameters from the count

    """Count the number of trainable parameters in a model."""
    # return sum(p.numel() for p in model.parameters())
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not re.match(r'^(model\.embed_tokens|lm_head)\.', name):
                total_params += param.numel()
            else:
                if do_print:
                    print(f"  Skipping parameters in {name}")

    all_params = sum(p.numel() for p in model.parameters())
    return total_params, all_params
