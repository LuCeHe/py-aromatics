import os, socket, base64


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

    from huggingface_hub import login

    # detect internet connection
    if connected_to_internet():
        login(token=token)

    # set the environment variable
    os.environ['HF_AUTH_TOKEN'] = token
    os.system(f"export HF_AUTH_TOKEN={token}")
    return token


def get_tokenizer(model_id, notes, save_dir=None, max_seq_length=None):
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

    if not max_seq_length is None:
        tokenizer.model_max_length = max_seq_length
        tokenizer.max_length = max_seq_length

    return tokenizer


def get_pretrained_model(model_id='gpt2', save_dir=None, return_path=False):
    if save_dir is None:
        raise ValueError("save_dir must be specified")

    import torch
    from transformers import AutoModelForCausalLM

    model_path = os.path.join(save_dir, model_id.replace('/', '-') + '-model')

    if return_path:
        return model_path

    more_kwargs = {'device_map': "auto", 'offload_buffers': True}
    if 'gemma' in model_id.lower():
        more_kwargs = {'attn_implementation': 'eager'}
    if not os.path.exists(model_path):
        print('Downloading Model')
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **more_kwargs)
        model.save_pretrained(model_path)
    else:
        print('Loading Model')
        model = AutoModelForCausalLM.from_pretrained(model_path, **more_kwargs)

    return model
