import tensorflow as tf



def exponential_annealing(epoch, epochs, value):
    if epoch == 0:
        new_value = 0
    elif epoch > epochs / 2:
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


hard_annealing=lambda epoch, epochs, value: 0 if epoch < epochs / 2 else 1


def get_annealing_schedule(annealing_schedule):
    if annealing_schedule in ['probabilistic_exponential_annealing', 'pea']:
        return probabilistic_exponential_annealing
    elif annealing_schedule in ['exponential_annealing', 'ea']:
        return exponential_annealing
    elif annealing_schedule in ['hard_annealing', 'ha']:
        return hard_annealing
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
                 annealing_schedule=lambda epoch, value: 1,
                 ):
        super().__init__()
        self.epochs = epochs
        self.variables_to_anneal = variables_to_anneal
        self.annealing_schedule = annealing_schedule if not isinstance(annealing_schedule, str) \
            else get_annealing_schedule(annealing_schedule)

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

        for w in self.annealing_weights:
            v = tf.keras.backend.get_value(w)
            tf.keras.backend.set_value(w, self.annealing_schedule(self.epoch, self.epochs, v))