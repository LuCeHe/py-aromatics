import numpy as np

from GenericTools.language_tools.unpadding import pad_sequences


def random_indices(vocab_size, batch_size=4, maxlen=7, padded=True, pad_idx=0, padding='pre'):
    non_pad_words = list(range(vocab_size))
    non_pad_words.remove(pad_idx)
    questions = []
    for _ in range(batch_size):
        sentence_length = np.random.choice(maxlen)
        randomQ = np.random.choice(non_pad_words, sentence_length)
        questions.append(randomQ)

    if padded:
        questions = pad_sequences(questions, value=pad_idx, maxlen=maxlen, padding=padding)
    return questions


def random_sequences_and_points(batch_size=3,
                                lat_dim=4,
                                maxlen=6,
                                repeated=False,
                                vocab_size=2,
                                hyperplane=False):
    if not repeated:
        questions = []
        points = np.random.rand(batch_size, lat_dim)
        for _ in range(batch_size):
            sentence_length = maxlen  # np.random.choice(maxlen)
            randomQ = np.random.choice(vocab_size, sentence_length)  # + 1
            # EOS = (vocab_size+1)*np.ones(1)
            # randomQ = np.concatenate((randomQ, EOS))
            questions.append(randomQ)
    else:
        point = np.random.rand(1, lat_dim)
        sentence_length = maxlen  # np.random.choice(maxlen)
        question = np.random.choice(vocab_size, sentence_length)  # + 1
        question = np.expand_dims(question, axis=0)
        points = np.repeat(point, repeats=[batch_size], axis=0)
        questions = np.repeat(question, repeats=[batch_size], axis=0)

    padded_questions = pad_sequences(questions)

    if hyperplane:
        point_1 = np.random.rand(1, 1)
        point_1 = np.repeat(point_1, repeats=[batch_size], axis=0)
        point_2 = np.random.rand(batch_size, lat_dim - 1)
        points = np.concatenate([point_1, point_2], axis=1)

    return padded_questions, points
