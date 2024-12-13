import numpy as np
import torch


def norm_spikegram(spikegrams: np.ndarray, arch: str = "fc", threshold=3e-3, num_channels=700, max_abs_val=0.05):
    # Select non zero values
    sample_idx, neuron_idx = np.nonzero(spikegrams)
    # Eliminate the interval ]-threshold, threshold[ because there is no non zero values in it
    spikegrams[sample_idx, neuron_idx] = (
            spikegrams[sample_idx, neuron_idx]
            - np.sign(spikegrams[sample_idx, neuron_idx]) * threshold
    )
    # Normalize values between 0 and 1
    spikegrams[sample_idx, neuron_idx] = (spikegrams[sample_idx, neuron_idx] - (max_abs_val - threshold)) / 2 / (
            max_abs_val - threshold)

    if arch == "cnn":
        spikegrams = np.reshape(spikegrams, (len(spikegrams), num_channels, -1))
    # We should return if arch is cnn
    return spikegrams


# shape(mini_batch) = (batch_size, features)
def ttfs(mini_batch: torch.Tensor, absolute_duration: int = 20, arch: str = "fc"):
    if arch == "cnn":
        sample_id, channel_id, neuron_id = torch.nonzero(mini_batch,
                                                         as_tuple=True)
        time_of_spike = (
                                1 - torch.abs(mini_batch)
                        ) * absolute_duration  # The higher the value, the earlier the spike
        nb_of_samples, input_channels, nb_of_features = mini_batch.size()
        time_of_spike = time_of_spike.long()
        spike_train = torch.zeros(nb_of_samples, input_channels,
                                  nb_of_features, absolute_duration)
        # spikes: +1 for positive values and -1 for negative values
        spike_train[sample_id, channel_id, neuron_id,
                    time_of_spike[sample_id, channel_id,
                                  neuron_id]] = torch.sign(
            mini_batch[sample_id, channel_id,
                       neuron_id]).float()
    else:
        sample_id, neuron_id = torch.nonzero(mini_batch, as_tuple=True)
        time_of_spike = (1 - torch.abs(mini_batch)) * absolute_duration  # The higher the value, the earlier the spike
        nb_of_samples, nb_of_features = mini_batch.size()
        time_of_spike = time_of_spike.long()

        spike_train = torch.zeros(nb_of_samples, nb_of_features, absolute_duration)
        # spikes: +1 for positive values and -1 for negative values
        spike_train[sample_id, neuron_id,
                    time_of_spike[sample_id, neuron_id]] = torch.sign(
            mini_batch[sample_id, neuron_id]).float()
    return spike_train  # (batchsize, features, absolute duration) (x, 700*108, 20) (x, 256x2, 700)

# - is it 20x slower for RNNs/causal convolution?
# -
