import torch, random

inbatchdocs_modes = [
    'addone', 'addchunk', 'addtwochunks', 'addthreechunks', 'addnoisy', 'default'
]
inbatchdocs_modes_2_batch_size = [
    'addone', 'addchunk', 'addnoisy', 'default'
]


def add_one_doc(docs, addition, shuffle=True):
    # add target
    docs = torch.cat([docs, addition.unsqueeze(1)], dim=1)  # (b, b, t)

    # remove b=0 in axis=1
    docs = docs[:, 1:, :]

    # shuffle ax 1
    if shuffle:
        docs = docs[:, torch.randperm(docs.size(1)), :]
    return docs


def add_chunk(batch_copy):
    time_steps = batch_copy.size(1)
    batch_size = batch_copy.size(0)
    if batch_size > 2:
        shift = torch.randint(1, batch_size - 1, (1,)).item()
    else:
        shift = 1

    # pick random m, n in t
    m = torch.randint(0, time_steps - 1, (1,)).item()
    delta = torch.randint((time_steps - m) // 2, time_steps - m, (1,)).item()

    main_chunk = batch_copy[:, m:m + delta]

    shifted_x = batch_copy[torch.roll(torch.arange(batch_size), shifts=shift)]
    left = shifted_x[:, :m]
    right = shifted_x[:, m + delta:]

    # Concatenate to get final tensor (b, t)
    batch_copy = torch.cat([left, main_chunk, right], dim=1)
    return batch_copy


def build_in_batch_docs(batch, mode='addthreechunks', batch_first=True, vocab_size=None):
    device = batch.device

    if not batch_first:
        # move from (t, b) to (b, t)
        batch = batch.permute(1, 0)

    batch_size = batch.size(0)

    inbatchdocs_modes_ = inbatchdocs_modes_2_batch_size if batch_size < 2 else inbatchdocs_modes
    if mode == 'auto':
        mode = random.choice(inbatchdocs_modes_)

    if not mode in inbatchdocs_modes:
        raise ValueError(f'Unknown mode: {mode}. Available modes: {inbatchdocs_modes}')

    if batch_size == 1:
        docs = batch.unsqueeze(0)
        if not batch_first:
            # move from (1, b, t) to (1, t, b)
            docs = docs.permute(0, 2, 1)

        return docs

    docs = torch.stack([
        torch.cat([batch[:i], batch[i + 1:]], dim=0)
        for i in range(batch_size)
    ], dim=0)

    # shuffle ax 1
    docs = docs[:, torch.randperm(docs.size(1)), :]

    batch_copy = batch.clone()

    if mode == 'addchunk':
        # add a chunk of the target as a new document
        batch_copy = add_chunk(batch_copy)
        docs = add_one_doc(docs, batch_copy)

    if mode == 'addtwochunks':
        # add two chunks of the target as two new documents
        batch_copy = add_chunk(batch)
        docs = add_one_doc(docs, batch_copy, shuffle=False)
        batch_copy = add_chunk(batch)
        docs = add_one_doc(docs, batch_copy)

    if mode == 'addthreechunks':
        # add three chunks of the target as three new documents
        batch_copy = add_chunk(batch)
        docs = add_one_doc(docs, batch_copy, shuffle=False)
        batch_copy = add_chunk(batch)
        docs = add_one_doc(docs, batch_copy, shuffle=False)
        batch_copy = add_chunk(batch)
        docs = add_one_doc(docs, batch_copy)

    if 'addnoisy' == mode:
        # rand matrix of integers
        rand_matrix = torch.randint(0, vocab_size, batch_copy.shape)

        # 40% tokens from rand_matrix, the rest from batch_copy
        rand_mask = torch.rand(batch_copy.shape) < 0.4
        batch_copy = torch.where(rand_mask.to(device), rand_matrix.to(device), batch_copy)
        docs = add_one_doc(docs, batch_copy)

    if mode == 'addone':
        # add the target as a new document
        docs = add_one_doc(docs, batch_copy)

    # docs should have shape (n_docs, b, t) with n_docs=b-1, and now it is (b, n_docs, t)
    docs = docs.permute(1, 0, 2)

    if not batch_first:
        # move from (b, n_docs, t) to (n_docs, b, t)
        docs = docs.permute(0, 2, 1)

    return docs


if __name__ == '__main__':
    # Test the in_batch_docs function with different modes

    batch_size = 2
    time_steps = 7
    vocab_size = batch_size * time_steps
    batch_first = True
    n_attempts = 1

    if batch_first:
        # Use a deterministic batch for easy debugging
        batch = torch.arange(batch_size * time_steps).reshape(batch_size, time_steps).float()
    else:
        batch = torch.arange(batch_size * time_steps).reshape(time_steps, batch_size).float()

    print('shape of batch:', batch.shape)
    for _ in range(n_attempts):
        # Use a deterministic batch for easy debugging
        docs = build_in_batch_docs(batch=batch, mode='auto', batch_first=batch_first, vocab_size=vocab_size)

    for i in range(3):
        print('-' * 20)
        print(batch[i])
        print(docs[:, :, i])  # should be x[1:]
