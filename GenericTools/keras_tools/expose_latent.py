import tensorflow as tf
import sys

from GenericTools.stay_organized.utils import flaggedtry
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


def truer_split_model(model, pairs):
    lnames = [layer.name for layer in model.layers]
    split_layer_name = lnames[pairs[0]]
    last_layer_name = lnames[pairs[1]]
    # print(pairs, split_layer_name, last_layer_name)

    model2split = model

    # Determine the split point based on the 'on_head' argument.
    # tail_input = tf.keras.layers.Input(batch_shape=model2split.get_layer(split_layer_name).get_input_shape_at(0))
    tail_input = None
    all_inputs = {
        l.name: tf.keras.layers.Input(batch_shape=model2split.get_layer(l.name).get_input_shape_at(0))
        for l in model2split.layers if 'input' in l.name
    }

    layer_outputs = {}
    extra_tail_inputs = []
    extra_input_names = []

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
            nonlocal tail_input

            output_shape = layer.output_shape if not isinstance(layer.output_shape, list) else layer.output_shape[0]
            if isinstance(tail_input, list):
                if len(tail_input) > 1:
                    raise ValueError('This scenario hasnt been implemented!')
            tail_input = tf.keras.layers.Input(batch_shape=output_shape)

            out = tail_input
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

        # Get the output of connected layers in a recursive manner
        pl_outs = []
        for pl in prev_layers:
            plo = _find_backwards(pl)
            if isinstance(plo, tuple):
                plo = list(plo)
            try:
                pl_outs.extend(plo)
            except TypeError:
                pl_outs.append(plo)

        # Apply this layer on the collected outputs
        out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
        layer_outputs[layer.name] = out
        return out

    # tail_output = _find_backwards(model2split.layers[-1])
    tail_output = _find_backwards(model2split.get_layer(last_layer_name))

    names = ('head', 'tail')
    # Creating head and tail models
    head_model = tf.keras.models.Model(
        model2split.input,
        [model2split.get_layer(split_layer_name).output]
        + [model2split.get_layer(ln).output for ln in extra_input_names], name=names[0]
    )
    tail_model = tf.keras.models.Model([tail_input] + extra_tail_inputs, tail_output, name=names[1])

    return head_model, tail_model



def test_split_model():
    import numpy as np
    from alif_sg.neural_models.transformer_model import build_model
    import random

    n_tries = 10
    tryornot = True
    modid = 'transf'  # eff simple transf simple2

    pairs = None
    jump = None
    skip_in_layers, skip_out_layers = [], ['input']
    keep_in_layers = None
    keep_out_layers = None
    if modid == 'eff':
        bm = lambda: EfficientNetB0(
            include_top=False, weights=None, activation='relu',
            batch_normalization=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform',
            comments=''
        )

    elif modid == 'simple':
        bm = lambda: simple_model()
        # pairs with [x, y] all work

    elif modid == 'simple2':
        bm = lambda: simple_model_2()

    elif modid == 'transf':
        bm = lambda: build_model(
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

        keep_in_layers = ['embeddinglayer', 'identity_']
        keep_out_layers = ['identity_']
        # jump = 10
    else:
        raise ValueError

    model = bm()
    model.summary()
    lnames = [layer.name for layer in model.layers]
    del model

    keep_in_layers = lnames if keep_in_layers is None else keep_in_layers
    keep_out_layers = lnames if keep_out_layers is None else keep_out_layers

    inlnames = [
                   i for i, l in enumerate(lnames)
                   if not any([s in l for s in skip_in_layers])
                      and any([s in l for s in keep_in_layers])
               ][:-1]
    outlnames = [
                    i for i, l in enumerate(lnames)
                    if not any([s in l for s in skip_out_layers])
                       and any([s in l for s in keep_out_layers])
                ][1:]

    for i in range(n_tries):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        model = bm()

        if pairs is None:
            inp_l = np.random.choice(inlnames)
            outlist = [i for i in outlnames if i > inp_l]
            out_l = np.random.choice(outlist)

            pairs = [inp_l, out_l]

        if not jump is None:
            actual_jump = np.random.choice(list(range(2, jump + 1)))
            pairs = [pairs[0], pairs[0] + actual_jump]

        head_model, tail_model = None, None
        output = flaggedtry(lambda: truer_split_model(model, pairs), tryornot=tryornot)
        if not output is None:
            head_model, tail_model = output

            lnames = [layer.name for layer in model.layers]
            last_layer_name = lnames[pairs[1]]
            direct_tail = tf.keras.models.Model(model.input, model.get_layer(last_layer_name).output)

            input_shapes = [model.get_layer(l.name).get_input_shape_at(0)
                            for l in model.layers if 'input' in l.name]
            input_shapes = [tuple([s if not s is None else 10 for s in shape]) for shape in input_shapes]

            input_noise = [tf.random.normal(shape) for shape in input_shapes]
            two_stages_output = tail_model(head_model(input_noise))
            direct_output = direct_tail(input_noise)
            # compare if they produce the same tensor
            print('Is the output of the split model the same as non split?',
                  tf.reduce_all(tf.equal(two_stages_output, direct_output)).numpy())
            del head_model, tail_model, direct_tail, input_shapes, input_noise, two_stages_output, direct_output
        del model
        pairs = None

if __name__ == '__main__':
    test_split_model()
