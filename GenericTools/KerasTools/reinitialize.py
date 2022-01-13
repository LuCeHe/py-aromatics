import tensorflow as tf


def reset_weights(model, variables_to_reset='all'):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer': #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            print(variables_to_reset)
            print(var.name)
            if var is not None:
                if var.name in variables_to_reset:
                    print(var.name)
                    var.assign(initializer(var.shape, var.dtype))
                elif variables_to_reset == 'all':
                    var.assign(initializer(var.shape, var.dtype))
