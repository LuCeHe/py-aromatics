import os
import matplotlib.pyplot as plt

def plot_history(history, plot_filename, epochs):
    if epochs > 0:
        # plot training and validation losses
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.plot(history.history['loss'], label='train')
        ax.plot(history.history['val_loss'], label='val')
        ax.set_title('model loss')
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend()
        fig.savefig(plot_filename, bbox_inches='tight')