from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def saveplot(losshistory, train_state, issave=True, isplot=True, plotspath='', outputspath=''):
    if isplot:
        plot_loss_history(losshistory, plotspath)
        plot_best_state(train_state, plotspath)
        plt.show()

    if issave:
        save_loss_history(losshistory, outputspath + r"loss.dat")
        save_best_state(train_state, outputspath + r"train.dat", outputspath + r"test.dat")


def plot_loss_history(losshistory, plotspath=''):
    loss_train = np.sum(
        np.array(losshistory.loss_train) * losshistory.loss_weights, axis=1
    )
    loss_test = np.sum(
        np.array(losshistory.loss_test) * losshistory.loss_weights, axis=1
    )

    plt.figure()
    plt.semilogy(losshistory.steps, loss_train, label="Train loss")
    plt.semilogy(losshistory.steps, loss_test, label="Test loss")
    for i in range(len(losshistory.metrics_test[0])):
        plt.semilogy(
            losshistory.steps,
            np.array(losshistory.metrics_test)[:, i],
            label="Test metric",
        )
    plt.xlabel("# Steps")
    plt.legend()
    plt.savefig(plotspath + r'history.png')


def save_loss_history(losshistory, fname):
    print("Saving loss history to {} ...".format(fname))
    loss = np.hstack(
        (
            np.array(losshistory.steps)[:, None],
            np.array(losshistory.loss_train),
            np.array(losshistory.loss_test),
            np.array(losshistory.metrics_test),
        )
    )
    np.savetxt(fname, loss, header="step, loss_train, loss_test, metrics_test")


def plot_best_state(train_state, plotspath=''):
    X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()

    y_dim = best_y.shape[1]

    # Regression plot
    if X_test.shape[1] == 1:
        idx = np.argsort(X_test[:, 0])
        X = X_test[idx, 0]
        plt.figure()
        for i in range(y_dim):
            if y_train is not None:
                plt.plot(X_train[:, 0], y_train[:, i], "ok", label="Train")
            if y_test is not None:
                plt.plot(X, y_test[idx, i], "-k", label="True")
            plt.plot(X, best_y[idx, i], "--r", label="Prediction")
            if best_ystd is not None:
                plt.plot(
                    X, best_y[idx, i] + 2 * best_ystd[idx, i], "-b", label="95% CI"
                )
                plt.plot(X, best_y[idx, i] - 2 * best_ystd[idx, i], "-b")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.savefig(plotspath + r'Regression.png')

    elif X_test.shape[1] == 2:
        for i in range(y_dim):
            plt.figure()
            ax = plt.axes(projection=Axes3D.name)
            ax.plot3D(X_test[:, 0], X_test[:, 1], best_y[:, i], ".")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$y_{}$".format(i + 1))
            plt.savefig(plotspath + r'Regression_{}.png'.format(i))

    # Residual plot
    if y_test is not None:
        plt.figure()
        residual = y_test[:, 0] - best_y[:, 0]
        plt.plot(best_y[:, 0], residual, "o", zorder=1)
        plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles="dashed", zorder=2)
        plt.xlabel("Predicted")
        plt.ylabel("Residual = Observed - Predicted")
        plt.tight_layout()

        plt.savefig(plotspath + r'Residual.png')

    if best_ystd is not None:
        plt.figure()
        for i in range(y_dim):
            plt.plot(X_test[:, 0], best_ystd[:, i], "-b")
            plt.plot(
                X_train[:, 0],
                np.interp(X_train[:, 0], X_test[:, 0], best_ystd[:, i]),
                "ok",
            )
        plt.xlabel("x")
        plt.ylabel("std(y)")
        plt.savefig(plotspath + r'std.png')


def save_best_state(train_state, fname_train, fname_test):
    print("Saving training data to {} ...".format(fname_train))
    X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()
    if y_train is None:
        np.savetxt(fname_train, X_train, header="x")
    else:
        train = np.hstack((X_train, y_train))
        np.savetxt(fname_train, train, header="x, y")

    print("Saving test data to {} ...".format(fname_test))
    if y_test is None:
        test = np.hstack((X_test, best_y))
        if best_ystd is None:
            np.savetxt(fname_test, test, header="x, y_pred")
        else:
            test = np.hstack((test, best_ystd))
            np.savetxt(fname_test, test, header="x, y_pred, y_std")
    else:
        test = np.hstack((X_test, y_test, best_y))
        if best_ystd is None:
            np.savetxt(fname_test, test, header="x, y_true, y_pred")
        else:
            test = np.hstack((test, best_ystd))
            np.savetxt(fname_test, test, header="x, y_true, y_pred, y_std")
