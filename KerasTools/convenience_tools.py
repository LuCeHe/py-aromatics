import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_history(history, plot_filename, epochs):
    if epochs > 0:
        keys = history.history.keys()
        keys = [k for k in keys if not 'val' in k]
        fig, axs = plt.subplots(len(keys), figsize=(8, 8), sharex='all',
                                gridspec_kw={'hspace': 0})

        fig.suptitle(plot_filename)
        for k, ax in zip(keys, axs):
            # plot training and validation losses

            try:
                history.history['val_' + k]
                ax.plot(history.history[k], label='train ' + k)
                ax.plot(history.history['val_' + k], label='val ' + k)
            except:
                ax.plot(history.history[k], label=k)
            ax.set_ylabel(k)
            ax.legend()
        ax.set_xlabel('epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(plot_filename, bbox_inches='tight')
