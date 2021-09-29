import tensorflow as tf
import matplotlib.pyplot as plt


class DummyConstantSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, ):
        super(DummyConstantSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.initial_learning_rate * tf.ones_like(step)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
        }


class TransformerLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(TransformerLRSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class AddWarmUpToSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, scheduler, warmup_steps=4000, alpha=.001):
        super(AddWarmUpToSchedule, self).__init__()

        if isinstance(scheduler, float):
            self.warm_learning_rate = scheduler
        else:
            self.warm_learning_rate = scheduler.initial_learning_rate
        self.initial_learning_rate = alpha * self.warm_learning_rate
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.alpha = alpha

    def __call__(self, step):
        print('in: ', step)
        step = tf.cast(step, tf.float32)
        arg1 = step * (self.warm_learning_rate - self.initial_learning_rate) / (
            self.warmup_steps) + self.initial_learning_rate
        arg2 = self.scheduler(step)
        warmup_steps = self.warmup_steps * tf.ones_like(step)
        alpha = tf.cast(tf.math.argmin([step, warmup_steps]), tf.float32)
        slr = (1 - alpha) * arg1 + alpha * arg2
        return slr

    def get_config(self):
        return {
            "scheduler": self.scheduler,
            "warmup_steps": self.warmup_steps,
            'alpha': self.alpha,
        }


if __name__ == '__main__':
    d_model = 512
    total_epochs = 30 * 375
    lr = 1e-3
    learning_rate = DummyConstantSchedule(lr)
    # learning_rate = TransformerLRSchedule(d_model, total_epochs / 5)
    learning_rate = tf.keras.experimental.CosineDecay(lr, decay_steps=int(4 * total_epochs / 5), alpha=.5)
    # learning_rate = tf.keras.experimental.LinearCosineDecay(lr, decay_steps=int(2 * total_epochs / 3), alpha=.5)
    # learning_rate = tf.keras.experimental.CosineDecayRestarts(lr, first_decay_steps=int(total_epochs / 6.5), alpha=.1)
    # learning_rate = tf.keras.experimental.PolynomialDecay(lr, decay_steps=int(2 * total_epochs / 3), alpha=.1)
    # learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(lr, decay_steps=int(2 * total_epochs / 3), end_learning_rate=.1*lr)
    # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=int(2 * total_epochs / 3), decay_rate=.1*lr)
    learning_rate = AddWarmUpToSchedule(learning_rate, warmup_steps=total_epochs / 6)

    lrs = [learning_rate(i).numpy() for i in tf.range(total_epochs, dtype=tf.float32)]
    print(lrs)
    plt.plot(lrs)
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.ylim(0., 1.3 * lr)
    plt.show()
