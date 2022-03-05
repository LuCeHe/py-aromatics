from scipy.signal import savgol_filter
import numpy as np


def convergence_estimation(loss, epsilon=1e-6):
    if loss[0] == 0:
        loss = [l + epsilon for l in loss]

    if not len(loss) < 4:
        # smooth the signal
        window_length = int(np.ceil(len(loss) / 3) // 2 * 2 + 1)
        yhat = savgol_filter(loss, window_length, 2)

        # criterion of convergence 1. needs to detect an initial decay phase of at least 2/3
        c1 = (loss[0] - loss[-1]) / loss[0] < 2 / 3

        # criterion of convergence 2. needs to detect a final still period
        third_length = int(len(loss) / 3)
        last_variance = np.std(yhat[-third_length:])
        total_variance = np.std(yhat)
        c2 = last_variance / total_variance

        c = c1 * (1 - c2)
    else:
        c = 0
    return c
