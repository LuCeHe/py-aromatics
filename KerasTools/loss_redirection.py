from GenericTools.KerasTools.focal_loss import categorical_focal_loss
import tensorflow_addons as tfa

def get_loss(loss_name):
    if 'categorical_crossentropy' in loss_name:
        loss = 'categorical_crossentropy'
    elif 'categorical_focal_loss:' in loss_name:
        n_out = int(loss_name.split(':')[1])
        loss = categorical_focal_loss(alpha=[[1 / n_out] * n_out], gamma=2)
    elif 'contrastive_loss':
        loss = tfa.losses.ContrastiveLoss()
    else:
        raise NotImplementedError
    return loss