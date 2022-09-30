import tensorflow as tf


def expose_latent_model(original_model, exclude_layers=[], include_layers=[], idx=None, return_names = False):
    layer_names = []
    intermediate_layers = []
    if not isinstance(idx, list): idx = [idx]
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
                    layer_names.append(l.name)
                except Exception as e:
                    print(e)

    test_model = tf.keras.models.Model(original_model.input, intermediate_layers, name='test_model')

    if return_names:
        return test_model, layer_names
    else:
        return test_model



def split_model(model, pairs):
    lnames = [layer.name for layer in model.layers]

    input_shape = model.layers[pairs[0] + 1].input_shape[1:]
    premodel = tf.keras.models.Model(model.inputs, model.get_layer(lnames[pairs[0]]).output)

    DL_input = tf.keras.layers.Input(input_shape)
    DL_model = DL_input
    for layer in model.layers[pairs[0] + 1:pairs[1] + 1]:
        if isinstance(layer.input, list):
            break
        DL_model = layer(DL_model)
    intermodel = tf.keras.models.Model(inputs=DL_input, outputs=DL_model)

    return premodel, intermodel

