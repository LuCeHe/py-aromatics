import tensorflow as tf
import sys

from alif_sg.neural_models.modified_efficientnet import EfficientNetB0

sys.setrecursionlimit(100000)


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


def simple_model():
    # make a simple keras model
    input_shape = (2, 3, 4)
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)

    y = tf.keras.layers.Dense(64)(x)

    x = tf.keras.layers.concatenate([x, y])
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def simple_model_2():
    # make a simple keras model
    input_shape = (2, 3, 4)
    inputs = tf.keras.layers.Input(input_shape)

    inputs_2 = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    y = tf.keras.layers.Dense(64)(inputs_2)

    x = tf.keras.layers.concatenate([x, y])
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[inputs, inputs_2], outputs=outputs)
    return model


def test_1(model):
    pairs = [48, 143]
    # split_model(model, pairs)

    model2split = model

    # Determine the split point based on the 'on_head' argument.
    tail_input = tf.keras.layers.Input(batch_shape=model2split.get_layer(split_layer_name).get_input_shape_at(0))
    all_inputs = {
        l.name: tf.keras.layers.Input(batch_shape=model2split.get_layer(l.name).get_input_shape_at(0))
        for l in model2split.layers if 'input' in l.name
    }
    all_input_names = [l.name for l in model2split.layers if 'input' in l.name]
    print(all_input_names)

    layer_outputs = {}
    extra_tail_inputs = []
    extra_input_names = []

    count = 0

    def _find_backwards(layer):
        """
        Returns outputs of a layer by moving backward and
        finding outputs of previous layers until reaching split layer.
        directly inspired by the answer at the link below
        with some modifications and corrections. https://stackoverflow.com/a/56228514
        This is an internal function.
        """

        if layer.name in layer_outputs:
            return layer_outputs[layer.name]

        if layer.name == split_layer_name:
            out = layer(tail_input)
            layer_outputs[layer.name] = out
            return out

        if 'input' in layer.name:
            extra_tail_inputs.append(all_inputs[layer.name])
            extra_input_names.append(layer.name)

            # out = layer(all_inputs[layer.name])
            out = all_inputs[layer.name]
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
        print(prev_layers)

        # Get the output of connected layers in a recursive manner
        pl_outs = []
        for pl in prev_layers:
            plo = _find_backwards(pl)
            if isinstance(plo, tuple):
                plo = list(plo)

            print('==', plo)
            try:
                pl_outs.extend(plo)
            except TypeError:
                pl_outs.append(plo)

        # Apply this layer on the collected outputs
        print('---', pl_outs)
        out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
        layer_outputs[layer.name] = out
        return out

    # tail_output = _find_backwards(model2split.layers[-1])
    tail_output = _find_backwards(model2split.get_layer(last_layer_name))

    names = ('head', 'tail')
    # Creating head and tail models
    print([l.name for l in extra_tail_inputs])

    print(last_layer_name)
    print(model2split.get_layer(last_layer_name).inbound_nodes)
    print(model2split.get_layer(last_layer_name).inbound_nodes[0].inbound_layers)
    print(model2split.get_layer(last_layer_name).inbound_nodes[1].inbound_layers)

    # head_outputs = [
    #     model2split.get_layer(l.name).output
    #     for node in model2split.get_layer(last_layer_name).inbound_nodes[:1]
    #     for l in node.inbound_layers
    # ]
    head_model = tf.keras.models.Model(
        model2split.input,
        model2split.get_layer(last_layer_name).output
        + [model2split.get_layer(ln).output for ln in extra_input_names], name=names[0]
    )
    tail_model = tf.keras.models.Model([tail_input] + extra_tail_inputs, tail_output, name=names[1])

    direct_tail = tf.keras.models.Model(model2split.input, model2split.get_layer(last_layer_name).output)
    head_model.summary()
    tail_model.summary()

    input_shapes = [model2split.get_layer(l.name).get_input_shape_at(0) for l in model2split.layers if
                    'input' in l.name]
    input_shapes = [tuple([s if not s is None else 10 for s in shape]) for shape in input_shapes]
    print(input_shapes)
    # print(shape)
    input_noise = [tf.random.normal(shape) for shape in input_shapes]
    two_stages_output = tail_model.predict(head_model.predict(input_noise))
    direct_output = direct_tail.predict(input_noise)
    # compare if they produce the same tensor
    print(two_stages_output)
    print(direct_output)
    print(tf.reduce_all(tf.equal(two_stages_output, direct_output)).numpy())

    print('head_model')
    layer_names = [layer.name for layer in head_model.layers]
    print(layer_names)

    print('tail_model')
    layer_names = [layer.name for layer in tail_model.layers]
    print(layer_names)

    print('direct_tail')
    layer_names = [layer.name for layer in direct_tail.layers]
    print(layer_names)
    print('--')


if __name__ == '__main__':
    from alif_sg.neural_models.transformer_model import build_model

    modid = 'simple'  # eff simple transf

    if modid == 'eff':
        model = EfficientNetB0(
            include_top=False, weights=None, activation='relu',
            batch_normalization=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform',
            comments=''
        )

        split_layer_name = 'block7a_se_reshape'
        last_layer_name = 'block7a_se_excite'

    elif modid == 'simple':
        model = simple_model()

        split_layer_name = 'dense'
        last_layer_name = 'flatten'


    elif modid == 'trasf':
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

        split_layer_name = 'eidentity_3_1'
        last_layer_name = 'eidentity_3_3'
    else:
        raise ValueError

    model.summary()
    test_1(model)

    layer_names = [layer.name for layer in model.layers]
    print(layer_names)
