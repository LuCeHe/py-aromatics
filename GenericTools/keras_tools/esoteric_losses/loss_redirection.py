from GenericTools.keras_tools.esoteric_losses.focal_loss import categorical_focal_loss
import tensorflow_addons as tfa
import tensorflow as tf


def get_loss(loss_name):
    if 'categorical_crossentropy' == loss_name:
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.1,
            reduction="auto",
            name="categorical_crossentropy",
        )
    elif 'categorical_focal_loss:' in loss_name:
        n_out = int(loss_name.split(':')[1])
        loss = categorical_focal_loss(alpha=[[1 / n_out] * n_out], gamma=2)
    elif 'contrastive_loss' in loss_name:
        loss = tfa.losses.ContrastiveLoss()
    elif 'sparse_categorical_crossentropy' in loss_name:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    elif 'mse' in loss_name:
        loss = tf.keras.losses.MSE
    else:
        raise NotImplementedError
    return loss
