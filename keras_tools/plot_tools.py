import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def history_pick(k, v, min_epochs=0):

    o = None
    if isinstance(v, str):
        if 'array' in v:
            v = ''.join(v.splitlines())
            v = v.replace('array', '')
            v = v.replace('dtype=float32', '')
            v = re.sub(' +', ' ', v)
            v = v.replace('(', '').replace(')', '').replace('),', '')
            v = v.replace('[', '').replace(']', '')
            v = v.replace(' ,', ',').replace(',,', ',')
            v = f'[{v}]'.replace(', ]', ']')

        if v.startswith('[') and v.endswith(']'):
            if not 'None' in v and not len(v) == 2:
                v = [float(n) for n in v[1:-1].split(', ')]
            elif 'None' in v:
                v = '[Nones]'

    if isinstance(v, list):
        if not len(v) == 0 and not isinstance(v[0], str):
            if isinstance(v[-1], list):
                if len(v[-1]) == 0:
                    v[-1] = np.nan
                else:
                    v[-1] = v[-1][-1]

            if not np.isnan(v).all():
                o = [(k + ' ends', f'{round(v[0], 3)}/{round(v[-1], 3)}'), (k + ' initial', v[0]),
                     (k + ' final', v[-1]),
                     (k + ' mean', np.nanmean(v)), (k + ' min', np.nanmin(v)), (k + ' max', np.nanmax(v)),
                     (k + ' list', v), (k + ' len', len(v)),
                     (k + ' argmin', np.nanargmin(v)), (k + ' argmax', np.nanargmax(v)),
                     ]
                o = tuple(o)
            elif np.isnan(v).all():
                v = ['nans']
            else:
                v = str(v)

    if o is None:
        o = (k, v),

    return o


def plot_history(histories, epochs, plot_filename=None, method_names=None, show=False, bkg_color='white',
                 metrics_to_show=[], column_id=[], colors=None, figsize=None, ylims={}, vertical=True, legend=True):
    if not isinstance(histories, list): histories = [histories]
    if isinstance(histories[0], dict):
        old_histories = histories
        histories = []
        for h in old_histories:
            history = lambda x: None
            history.history = h
            histories.append(history)

    if not method_names is None:
        assert len(method_names) == len(histories)

    if epochs > 0:
        keys = []
        for h in histories:
            keys.extend(list(h.history.keys()))

        keys = sorted(list(set([k.replace('val_', '') for k in keys])))

        if metrics_to_show:
            keys = [k for k in keys if k in metrics_to_show]

        if vertical:
            n_columns = len(column_id) if not len(column_id) == 0 else 1
            n_rows = len(keys)
        else:
            n_columns = len(keys)
            n_rows = len(column_id) if not len(column_id) == 0 else 1

        if figsize is None:
            figsize = (20, 5) if n_columns > 1 else (5, 20)

        fig, axs = plt.subplots(n_rows, n_columns, figsize=figsize, gridspec_kw={'hspace': 0})
        axs = axs if type(axs) is np.ndarray else [axs]

        if colors is None:
            cm = plt.get_cmap('tab20')  # Reds tab20 gist_ncar
            colors = cm(np.linspace(1, 0, len(histories)))
            np.random.shuffle(colors)

        lines = []
        if method_names is None:
            method_names = [None] * len(histories)

        for history, c, m in zip(histories, colors, method_names):
            for i, k in enumerate(keys):

                column = [x in m for x in column_id].index(True) if column_id else 0

                ax = axs[(i, column) if column_id else i]
                ax.set_facecolor(bkg_color)

                # plot training and validation losses
                if k in history.history.keys():
                    try:
                        check_if_break = history.history['val_' + k]
                        line, = ax.plot(history.history[k], label='train ' + k, color=c)
                        ax.plot(history.history['val_' + k], label='val ' + k, color=c, linestyle='--')
                    except:
                        line, = ax.plot(history.history[k], label=k, color=c, linestyle='--')
                        if method_names is None:
                            ax.legend()

                if column == 0:
                    ax.set_ylabel(k.replace('_', '\n'))

                if k in ylims:
                    ax.set_ylim(*ylims[k])

            lines.append(line)
            ax.set_xlabel('epoch')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # fig.align_ylabels(axs[:, 0] if n_columns > 1 else axs[:])

        if not method_names is None and legend:
            ax.legend(lines, method_names)

        if not plot_filename is None:
            fig.savefig(plot_filename, bbox_inches='tight')

        if show:
            plt.show()

        return fig, axs


