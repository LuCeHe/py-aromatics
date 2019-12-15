import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def random_sequences_and_points(batchSize=3,
                                latDim=4,
                                max_senLen=6,
                                repeated=False,
                                vocabSize=2,
                                hyperplane=False):

    if not repeated:
        questions = []
        points = np.random.rand(batchSize, latDim)
        for _ in range(batchSize):
            sentence_length = max_senLen  # np.random.choice(max_senLen)
            randomQ = np.random.choice(vocabSize, sentence_length)  # + 1
            # EOS = (vocabSize+1)*np.ones(1)
            # randomQ = np.concatenate((randomQ, EOS))
            questions.append(randomQ)
    else:
        point = np.random.rand(1, latDim)
        sentence_length = max_senLen  # np.random.choice(max_senLen)
        question = np.random.choice(vocabSize, sentence_length)  # + 1
        question = np.expand_dims(question, axis=0)
        points = np.repeat(point, repeats=[batchSize], axis=0)
        questions = np.repeat(question, repeats=[batchSize], axis=0)

    padded_questions = pad_sequences(questions)

    if hyperplane:
        point_1 = np.random.rand(1, 1)
        point_1 = np.repeat(point_1, repeats=[batchSize], axis=0)
        point_2 = np.random.rand(batchSize, latDim - 1)
        points = np.concatenate([point_1, point_2], axis=1)

    return padded_questions, points
