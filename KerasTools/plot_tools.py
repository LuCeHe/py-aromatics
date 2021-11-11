import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


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

        keys = list(set(keys))
        keys = sorted([k for k in keys if not 'val' in k])

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


def TensorboardToNumpy(event_filename: str):
    # original:
    # https://stackoverflow.com/questions/47232779/how-to-extract-and-save-images-from-tensorboard-event-summary

    import tensorflow as tf
    # topic_counter = defaultdict(lambda: 0)

    serialized_examples = tf.data.TFRecordDataset(event_filename)
    for serialized_example in serialized_examples:
        print(serialized_example)
        # event = event_pb2.Event.FromString(serialized_example.numpy())
        # for v in event.summary.value:
        #     print(v)
        #     # if v.tag in image_tags:
        #     #
        #     #     if v.HasField('tensor'):  # event for images using tensor field
        #     #         s = v.tensor.string_val[2]  # first elements are W and H
        #     #
        #     #         tf_img = tf.image.decode_image(s)  # [H, W, C]
        #     #         np_img = tf_img.numpy()
        #     #
        #     #         topic_counter[v.tag] += 1
        #     #
        #     #         cnt = topic_counter[v.tag]
        #     #         tbi = TensorBoardImage(topic=v.tag, image=np_img, cnt=cnt)
        #     #
        #     #         yield tbi