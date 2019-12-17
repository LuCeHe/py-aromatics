import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


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
