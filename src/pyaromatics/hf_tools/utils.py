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
        print("Connected to the internet, logging in to Hugging Face Hub...")
        login(token=token)

    # set the environment variable
    os.environ['HF_AUTH_TOKEN'] = token
    os.system(f"export HF_AUTH_TOKEN={token}")
    return token
