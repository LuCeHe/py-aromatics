import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


# reduce_prod
def history_pick(k, v, min_epochs=0):
    # if isinstance(v,list):
    # v = np.array(v)
    if isinstance(v, list):
        if any([n in k for n in ['loss', 'perplexity', 'entropy', 'bpc']]):
            o = np.nanmin(v[min_epochs:])
        elif any([n in k for n in ['acc']]):
            o = np.nanmax(v[min_epochs:])
        else:
            o = f'{round(v[0], 3)}/{round(v[-1], 3)}'
            o = ((k, o), (k + '_initial', v[0]), (k + '_final', v[-1]))

        if not isinstance(o, tuple):
            o = (k, o),

    else:
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


def TensorboardToNumpy(event_filename: str, id_selection='', field='histo'):
    # original:
    # https://stackoverflow.com/questions/47232779/how-to-extract-and-save-images-from-tensorboard-event-summary
    assert field in ['image', 'histo', 'audio', 'tensor', 'simple_value']
    import tensorflow as tf
    from tensorboard.compat.proto import event_pb2

    # topic_counter = defaultdict(lambda: 0)

    serialized_examples = tf.data.TFRecordDataset(event_filename)
    means = {}
    stds = {}
    for serialized_example in serialized_examples:

        event = event_pb2.Event.FromString(serialized_example.numpy())

        if event.step not in means.keys():
            means[event.step] = {}
            stds[event.step] = {}

        for v in event.summary.value:
            if id_selection in v.tag and v.HasField(field):
                item = v.__getattribute__(field)
                mean = item.sum / item.num
                ex2 = item.sum_squares / item.num
                variance = ex2 - mean ** 2

                means[event.step].update({v.tag: mean})
                stds[event.step][v.tag] = np.sqrt(variance)

    means = {k: means[k] for k in sorted(means.keys())}
    stds = {k: stds[k] for k in sorted(stds.keys())}
    return means, stds


def TensorboardToNumpy_new(event_filename: str, id_selection='', field='histo'):
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    # Just in case, PATH_OF_FILE is the path of the file, not the folder
    loader = EventFileLoader(event_filename)

    # Where to store values
    wtimes, steps, actions = [], [], []
    for event in loader.Load():
        wtime = event.wall_time
        step = event.step
        if len(event.summary.value) > 0:
            summary = event.summary.value[0]
            if id_selection in summary.tag:
                print('-' * 50)
                print(step, wtime)
                print(summary.tag)

                # if summary.tag == HISTOGRAM_TAG:
                wtimes += [wtime] * int(summary.histo.num)
                steps += [step] * int(summary.histo.num)
                print(summary)
                print(summary.DESCRIPTOR)
                print(summary.__dir__())
                print(summary.histo.__dir__())
                print(summary.histo)
                print(summary.tensor)
                print(summary.histo.bucket)
                for num, val in zip(summary.histo.bucket, summary.histo.bucket_limit):
                    actions += [val] * int(num)
                    print(num)

                print(actions)
