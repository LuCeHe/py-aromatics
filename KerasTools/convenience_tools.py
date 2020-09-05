import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def plot_history(histories, plot_filename, epochs, method_names=None):
    if not isinstance(histories, list): histories = [histories]

    if not method_names is None:
        assert len(method_names) == len(histories)

    if epochs > 0:
        keys = histories[0].history.keys()
        keys = [k for k in keys if not 'val' in k]
        fig, axs = plt.subplots(len(keys), figsize=(8, 8), sharex='all',
                                gridspec_kw={'hspace': 0})

        fig.suptitle(plot_filename)

        colors = plt.cm.gist_ncar(np.linspace(0, 1, len(histories)))
        np.random.shuffle(colors)

        lines = []
        for history, c in zip(histories, colors):
            for k, ax in zip(keys, axs):
                # plot training and validation losses

                try:
                    history.history['val_' + k]
                    line, = ax.plot(history.history[k], label='train ' + k, color=c)
                    ax.plot(history.history['val_' + k], label='val ' + k, color=c, linestyle='--')
                    if k == keys[0]:
                        lines.append(line)
                except:
                    ax.plot(history.history[k], label=k)
                    if method_names is None:
                        ax.legend()
                ax.set_ylabel(k)
            ax.set_xlabel('epoch')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))


        if not method_names is None:
            ax.legend(lines, method_names)

        fig.savefig(plot_filename, bbox_inches='tight')
