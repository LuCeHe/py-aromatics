import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def standardize_dataset_names(initial_name):
    name = ''
    if initial_name == 'ptb':
        name = 'PTB'
    elif initial_name == 'heidelberg':
        name = 'SHD'
    elif initial_name == '_s_mnist':
        name = 'sMNIST'
    elif initial_name == '_ps_mnist':
        name = 'psMNIST'
    return name

def plot_history(histories, plot_filename, epochs, method_names=None, save=True, show=False, bkg_color='white',
                 keys_to_skip=[], column_id=[]):
    if not isinstance(histories, list): histories = [histories]

    if not method_names is None:
        assert len(method_names) == len(histories)

    if epochs > 0:
        keys = []
        for h in histories:
            keys.extend(list(h.history.keys()))

        keys = list(set(keys))
        keys = [k for k in keys if not 'val' in k]

        if keys_to_skip:
            keys = [k for k in keys if not k in keys_to_skip]

        print(keys)
        n_columns = len(column_id) if not len(column_id) == 0 else 1
        fig, axs = plt.subplots(len(keys), n_columns, figsize=(8, 8), gridspec_kw={'hspace': 0})
        axs = axs if type(axs) is np.ndarray else [axs]

        # fig.suptitle(plot_filename)

        # colors = plt.cm.gist_ncar(np.linspace(0, 1, len(histories)))
        cm = plt.get_cmap('Reds')
        colors = cm(np.linspace(1, 0, len(histories)))
        np.random.shuffle(colors)

        lines = []

        if method_names is None:
            method_names = [None]*len(histories)
        for history, c, m in zip(histories, colors, method_names):
            print(m)
            for i, k in enumerate(keys):

                column = [x in m for x in column_id].index(True) if column_id else 0

                ax = axs[(i, column) if column_id else i]
                ax.set_facecolor(bkg_color)
                if i == 0 and column_id:
                    ax.set_title(standardize_dataset_names(column_id[column]))

                # plot training and validation losses
                if k in history.history.keys():
                    try:
                        history.history['val_' + k]
                        line, = ax.plot(history.history[k], label='train ' + k, color=c)
                        ax.plot(history.history['val_' + k], label='val ' + k, color=c, linestyle='--')
                        if k == keys[0]:
                            lines.append(line)
                    except:
                        ax.plot(history.history[k], label=k, color=c, linestyle='--')
                        if method_names is None:
                            ax.legend()
                if column == 0:
                    ax.set_ylabel(k.replace('_', '\n'))

            ax.set_xlabel('epoch')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if not method_names is None:
            # ax.legend(lines, method_names)
            pass

        if save:
            fig.savefig(plot_filename, bbox_inches='tight')

        if show:
            plt.show()
