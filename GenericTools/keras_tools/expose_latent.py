import tensorflow as tf


def expose_latent_model(original_model, exclude_layers=[], include_layers=[], idx=None, return_names=False):
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
    # FIXME: make it work for complex networks in the intermodel
    lnames = [layer.name for layer in model.layers]

    input_shape = model.get_layer(lnames[pairs[0] + 1]).input_shape[1:]

    DL_input = tf.keras.layers.Input(input_shape)
    DL_model = DL_input
    layers = {}
    inp_name = model.layers[pairs[0]].name
    out_name = ''
    for layer in model.layers[pairs[0] + 1:pairs[1] + 1]:
        layers[layer.name] = layer

        if isinstance(layer.input, list):
            # print('letssee')
            # print([l.name for l in layer.input])
            # print(layers.keys())
            break
        DL_model = layer(DL_model)

    premodel = tf.keras.models.Model(model.inputs, model.get_layer(lnames[pairs[0]]).output)
    intermodel = tf.keras.models.Model(inputs=DL_input, outputs=DL_model)

    return premodel, intermodel


if __name__ == '__main__':
    from alif_sg.neural_models.transformer_model import build_model

    model = build_model(
        inputs_timesteps=3,
        target_timesteps=4,
        inputs_vocab_size=2,
        target_vocab_size=2,
        encoder_count=4,
        decoder_count=4,
        attention_head_count=2,
        d_model=2,
        d_point_wise_ff=2,
        dropout_prob=.1,
        activation='relu',
        comments='',
    )

    pairs = [48, 143]
    # split_model(model, pairs)

    split_layer_name = 'eidentity_3_3'
    model2split = model

    # Determine the split point based on the 'on_head' argument.
    tail_input = tf.keras.layers.Input(batch_shape=model2split.get_layer(split_layer_name).get_input_shape_at(0))

    layer_outputs = {}


    def _find_backwards(layer):
        """
        Returns outputs of a layer by moving backward and
        finding outputs of previous layers until reaching split layer.
        directly inspired by the answer at the link below
        with some modifications and corrections. https://stackoverflow.com/a/56228514
        This is an internal function.
        """

        print(layer.name)

        if layer.name in layer_outputs:
            return layer_outputs[layer.name]

        if layer.name == split_layer_name:
            out = layer(tail_input)
            layer_outputs[layer.name] = out
            return out

        # Find all the connected layers which this layer consumes their output
        prev_layers = []
        for node in layer.inbound_nodes:
            try:
                # If number of inbound layers > 1
                prev_layers.extend(node.inbound_layers)
            except TypeError:
                # If number of inbound layers == 1
                prev_layers.append(node.inbound_layers)

        # Get the output of connected layers in a recursive manner
        pl_outs = []
        for pl in prev_layers:
            plo = _find_backwards(pl)
            try:
                pl_outs.extend([plo])
            except TypeError:
                pl_outs.append([plo])

        # Apply this layer on the collected outputs
        print('---', pl_outs)
        out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
        layer_outputs[layer.name] = out
        return out


    tail_output = _find_backwards(model2split.layers[-1])

    names = ('head', 'tail')
    # Creating head and tail models
    head_model = tf.keras.models.Model(model2split.input, model2split.get_layer(split_layer_name).output, name=names[0])
    tail_model = tf.keras.models.Model(tail_input, tail_output, name=names[1])
    head_model.summary()
    tail_model.summary()
