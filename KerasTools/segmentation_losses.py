import tensorflow as tf

from GenericTools.KerasTools.advanced_losses import surface_loss

try:
    from segmentation_models.base import Loss
except:
    from segmentation_models.segmentation_models.base import Loss


class SurfaceLoss(Loss):

    def __init__(self):
        super().__init__(name='surface_loss')

    def __call__(self, gt, pr):
        return surface_loss(gt, pr)


class SNRLoss(Loss):

    def __init__(self, epsilon=1e-5):
        super().__init__(name='snr_loss')
        self.epsilon = epsilon

    def __call__(self, gt, pr):
        S = gt
        N = gt - pr + self.epsilon
        SNR = tf.math.truediv(S, N)
        return SNR
