import tensorflow as tf


def expose_latent_model(original_model, exclude_layers=[], include_layers=[], idx=None):
    intermediate_layers = []
    if not isinstance(idx ,list): idx=[idx]
    for l in original_model.layers:
        if (not l.name in exclude_layers) and all([not el in l.name for el in exclude_layers]):
            if (l.name in include_layers) or all([el in l.name for el in include_layers]):
                try:
                    if isinstance(l.output_shape[0], int):
                        layer = original_model.get_layer(l.name).output
                        intermediate_layers.append(layer)
                    else:
                        layer = original_model.get_layer(l.name).output
                        if l.output_shape[0] == None:
                            layer = (layer,)
                        if idx[0] is None:
                            intermediate_layers.extend(layer)
                        else:
                            for i in idx:
                                intermediate_layers.append(layer[i])
                except Exception as e:
                    print(e)

    test_model = tf.keras.models.Model(original_model.input, intermediate_layers, name='test_model')
    return test_model