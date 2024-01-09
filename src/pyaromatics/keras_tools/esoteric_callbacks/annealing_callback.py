import tensorflow as tf


def linear_annealing(epoch, epochs, value):
    r_f = 3 / 5
    r_i = 1 / 5
    if epoch < r_i * epochs:
        new_value = 0
    elif epoch > r_f * epochs:
        new_value = 1
    else:
        new_value = 1 / (r_f * epochs - r_i * epochs) * (epoch - r_i * epochs)
    return new_value


def exponential_annealing(epoch, epochs, value):
    if epoch < epochs / 5:
        new_value = 0
    elif epoch > 3 * epochs / 4:
        new_value = 1
    else:
        new_value = 1 - .6 * (1 - value)

    return new_value


def probabilistic_exponential_annealing(epoch, epochs, value):
    if epoch == 0:
        probability_of_1 = 0
    elif epoch > epochs / 2:
        probability_of_1 = 1
    else:
        probability_of_1 = 1 - tf.exp(-epoch / epochs * 5)
    new_value = tf.random.uniform(()) < probability_of_1

    return new_value


hard_annealing = lambda epoch, epochs, value: 0 if epoch < epochs / 2 else 1
inverse_hard_annealing = lambda epoch, epochs, value: 1 if epoch < epochs / 2 else 0

cosinusoidal_annealing = lambda epoch, epochs, value: 1 / 2 * (1 + tf.cos(value*epoch * 3.14159*2))

add_1 = lambda epoch, epochs, value: value+1
def get_annealing_schedule(annealing_schedule):
    if annealing_schedule in ['probabilistic_exponential_annealing', 'pea']:
        return probabilistic_exponential_annealing
    elif annealing_schedule in ['exponential_annealing', 'ea']:
        return exponential_annealing
    elif annealing_schedule in ['linear_annealing', 'la']:
        return linear_annealing
    elif annealing_schedule in ['hard_annealing', 'ha', 'switch_on']:
        return hard_annealing
    elif annealing_schedule in ['inverse_hard_annealing', 'iha', 'switch_off']:
        return inverse_hard_annealing
    elif annealing_schedule in ['cos']:
        return cosinusoidal_annealing
    elif annealing_schedule in ['+1']:
        return add_1
    else:
        raise NotImplementedError


class AnnealingCallback(tf.keras.callbacks.Callback):
    """

    # exponential decay
    annealing_schedule=lambda epoch, value: .6 * value

    # hard switch
    annealing_schedule=lambda epoch, value: 0 if epoch < final_epochs / 2 else 1

    """

    def __init__(self,
                 epochs,
                 variables_to_anneal=[],
                 annealing_schedule=[lambda epoch, value: 1],
                 ):
        super().__init__()
        self.epochs = epochs
        self.variables_to_anneal = variables_to_anneal
        annealing_schedule = annealing_schedule if isinstance(annealing_schedule, list) else [annealing_schedule]

        self.annealing_schedule = [ans if not isinstance(ans, str) else get_annealing_schedule(ans)
                                   for ans in annealing_schedule]

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
        self.model = model
        self.annealing_weights = []
        for va in self.variables_to_anneal:
            for layer in self.model.layers:
                for weight in layer.weights:
                    if va in weight.name:
                        self.annealing_weights.append(weight)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        # print('inside here', len(self.annealing_weights), len(self.annealing_schedule))
        if len(self.annealing_schedule) == 1 and not len(self.annealing_weights) == 1:
            self.annealing_schedule = [self.annealing_schedule[0] for _ in self.annealing_weights]

        for w, ans in zip(self.annealing_weights, self.annealing_schedule):
            v = tf.keras.backend.get_value(w)
            tf.keras.backend.set_value(w, ans(self.epoch, self.epochs, v))




class EpochAnnealingCallback(tf.keras.callbacks.Callback):
    """

    # exponential decay
    annealing_schedule=lambda epoch, value: .6 * value

    # hard switch
    annealing_schedule=lambda epoch, value: 0 if epoch < final_epochs / 2 else 1

    """

    def __init__(self,
                 epochs,
                 variables_to_anneal=[],
                 annealing_schedule=[lambda epoch, value: 1],
                 ):
        super().__init__()
        self.epochs = epochs
        self.variables_to_anneal = variables_to_anneal
        annealing_schedule = annealing_schedule if isinstance(annealing_schedule, list) else [annealing_schedule]

        self.annealing_schedule = [ans if not isinstance(ans, str) else get_annealing_schedule(ans)
                                   for ans in annealing_schedule]

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
        self.model = model
        self.annealing_weights = []
        for va in self.variables_to_anneal:
            for layer in self.model.layers:
                for weight in layer.weights:
                    if va in weight.name:
                        self.annealing_weights.append(weight)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

        # print('inside here', len(self.annealing_weights), len(self.annealing_schedule))
        if len(self.annealing_schedule) == 1 and not len(self.annealing_weights) == 1:
            self.annealing_schedule = [self.annealing_schedule[0] for _ in self.annealing_weights]

        for w, ans in zip(self.annealing_weights, self.annealing_schedule):
            v = tf.keras.backend.get_value(w)
            tf.keras.backend.set_value(w, ans(self.epoch, self.epochs, v))
