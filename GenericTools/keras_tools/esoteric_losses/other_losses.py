from tensorflow.keras.losses import mean_absolute_error


def maechannel(channel):
    def msac(y_true, y_pred):
        yt = y_true[..., channel]
        yp = y_pred[..., channel]
        return mean_absolute_error(yt, yp)
    return msac