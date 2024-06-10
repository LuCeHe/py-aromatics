from keras.losses import mean_absolute_error

from pyaromatics.stay_organized.utils import rename


def maechannel(channel):
    @rename('maec{}'.format(channel))
    def maec(y_true, y_pred):
        yt = y_true[..., channel]
        yp = y_pred[..., channel]
        return mean_absolute_error(yt, yp)

    return maec

