import tensorflow as tf
import math

from GenericTools.KerasTools.esoteric_layers.random_switch import RandomSwitch
from GenericTools.PlotTools.mpl_tools import load_plot_settings
from GenericTools.StayOrganizedTools.utils import str2val


@tf.custom_gradient
def NelsonSpike_old(delta_b, delta_p, l_p_z):
    z = binary_forward(l_p_z)
    hard_b = tf.stop_gradient(gate(z))

    def grad(dy_z, dy_b):
        d_nelson = delta_b / delta_p * dy_b  # dy[1]
        return d_nelson, tf.zeros_like(delta_p), tf.zeros_like(l_p_z)

    return [z, hard_b], grad


@tf.custom_gradient
def NelsonSpike(v_sc, old_v_sc, thr):
    z_ = tf.cast(tf.greater(v_sc, 0.), dtype=tf.float32)

    def grad(dy):
        d_nelson = thr / (v_sc - old_v_sc) * dy  # dy[1]
        return d_nelson, tf.zeros_like(old_v_sc), tf.zeros_like(thr)

    return tf.identity(z_, name="nelsonSpikeFunction"), grad


@tf.custom_gradient
def SpikeFunction(v_scaled, dampening, sharpness):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled * sharpness), 0) * dampening
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(sharpness)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def ExpSpikeFunction(v_scaled, dampening, sharpness):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        dz_dv_scaled = tf.math.exp(-tf.abs(2 * v_scaled * sharpness)) * dampening
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(sharpness)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def FastSigmoidSpikeFunction(v_scaled, dampening, sharpness):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        xabs = tf.abs(2. * v_scaled * sharpness)
        dz_dv_scaled = 1 / (1 + xabs) ** 2
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(sharpness)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def NTailSpikeFunction(v_scaled, dampening, sharpness, tail):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        xabs = tf.abs(v_scaled * sharpness)
        factor = (tail - 1) / 2
        dz_dv_scaled = 1 / (1 + xabs / factor) ** tail
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(sharpness), tf.zeros_like(tail)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def CappedSkipSpikeFunction(v_scaled, dampening, sharpness):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        cap = tf.cast(tf.less(tf.abs(2 * sharpness * v_scaled), 1), dtype=tf.float32)
        return [dy * cap, tf.zeros_like(dampening), tf.zeros_like(sharpness)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def SpikeFunctionDamp(v_scaled, dampening):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0) * dampening
        return [
            dy * dz_dv_scaled,
            tf.reduce_mean(dy * dz_dv_scaled) * tf.ones_like(dampening)
        ]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def SpikeFunctionSigmoid(v_scaled, dampening, sharpness):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.keras.backend.floatx())

    def grad(dy):  # best m found 25
        x = 4 * v_scaled * sharpness
        dz_dv_scaled = 4 * tf.math.sigmoid(x) * (1 - tf.math.sigmoid(x)) * dampening
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(sharpness)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def SpikeFunctionGauss(v_scaled, dampening, sharpness):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.keras.backend.floatx())

    def grad(dy):  # 10 best multiplicative factors found
        dz_dv_scaled = tf.exp(-math.pi * tf.pow(v_scaled * sharpness, 2)) * dampening
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(sharpness)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def SpikeFunctionCauchy(v_scaled, dampening, sharpness):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.keras.backend.floatx())

    def grad(dy):  # 20 best multiplicative factors found
        dz_dv_scaled = 1 / (1 + tf.pow(math.pi * v_scaled * sharpness, 2)) * dampening
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(sharpness)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def SpikeFunctionDeltaDirac(v_scaled, dampening):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.keras.backend.floatx())

    def grad(dy):
        dz_dv_scaled = tf.cast(tf.math.equal(v_scaled, 0), tf.keras.backend.floatx())
        # dz_dv_scaled = 0
        return [dy * dz_dv_scaled, tf.zeros_like(dampening)]

    return tf.identity(z_, name="SpikeFunctionDeltaDirac"), grad


@tf.custom_gradient
def isiSpikeFunction(v_scaled, last_spike_distance):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        return [dy * last_spike_distance, tf.zeros_like(last_spike_distance)]

    return tf.identity(z_, name="isiSpikeFunction"), grad


@tf.custom_gradient
def isi2SpikeFunction(v_scaled, last_spike_distance):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0) * .3
        return [dy * last_spike_distance * dz_dv_scaled, tf.zeros_like(last_spike_distance)]

    return tf.identity(z_, name="isi2SpikeFunction"), grad


@tf.custom_gradient
def SpikeFunction_new(v_scaled, dampening, spike_dropout):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0) * dampening
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(spike_dropout)]

    n_neurons = tf.shape(z_)[1]
    batch_size = tf.shape(z_)[0]
    p = tf.tile([[spike_dropout, 1 - spike_dropout]], [batch_size, 1])
    mask = tf.cast(tf.random.categorical(tf.math.log(p), n_neurons), dtype=tf.float32)
    return tf.identity(mask * z_, name="SpikeFunction"), grad


# new derivatives ================================================================

@tf.custom_gradient
def idSpikeFunction(v_scaled, dampening):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        return [dy, tf.zeros_like(dampening)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def strangeSpikeFunction(v_scaled, dampening):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        dz_dv_scaled = 1 / (1 + tf.exp(-1 / v_scaled))
        return [dy * dz_dv_scaled, tf.zeros_like(dampening)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def switchSpikeFunction(v_scaled, dampening):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        aa_zeros = tf.cast(tf.random.categorical(tf.math.log([[.8, .2]]), 1), dtype=tf.float32)

        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0) * dampening
        dz_dv = (1 - aa_zeros) * dy * dz_dv_scaled + aa_zeros * dy
        return [dz_dv, tf.zeros_like(dampening)]

    return tf.identity(z_, name="SpikeFunction"), grad


def ChoosePseudoHeaviside(v_sc, config='', sharpness=1, dampening=1):
    sharpness = str2val(config, 'sharpn', float, default=sharpness)
    dampening = str2val(config, 'dampf', float, default=dampening)

    if 'gaussianpseudod' in config:
        z = SpikeFunctionGauss(v_sc, dampening, sharpness)

    elif 'cauchypseudod' in config:
        z = SpikeFunctionCauchy(v_sc, dampening, sharpness)

    elif 'originalpseudod' in config:
        z = SpikeFunction(v_sc, dampening, sharpness)

    elif 'sigmoidalpseudod' in config:
        z = SpikeFunctionSigmoid(v_sc, dampening, sharpness)

    elif 'exponentialpseudod' in config:
        z = ExpSpikeFunction(v_sc, dampening, sharpness)

    elif 'cappedskippseudod' in config:
        z = CappedSkipSpikeFunction(v_sc, dampening, sharpness)

    elif 'fastsigmoidpseudod' in config:
        z = FastSigmoidSpikeFunction(v_sc, dampening, sharpness)

    elif 'ntailpseudod' in config:
        tail = str2val(config, 'tailvalue', float, default=1.1)
        z = NTailSpikeFunction(v_sc, dampening, sharpness, tail)

    elif 'reluspike' in config:
        z = dampening * tf.nn.relu(sharpness * v_sc)

    elif 'ssnu' in config:
        z = dampening * tf.nn.sigmoid(sharpness * v_sc)

    elif 'geluspike' in config:
        z = dampening * tf.nn.gelu(sharpness * v_sc)

    elif 'softplusspike' in config:
        z = dampening * tf.math.softplus(sharpness * v_sc)

    else:
        z = FastSigmoidSpikeFunction(v_sc, dampening, sharpness)

    return z


class SurrogatedStep(tf.keras.layers.Layer):
    """
    Layer for a Heaviside with a surrogate gradient for the backward pass
    """

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, string_config='', dampening=1, sharpness=1, **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(string_config=string_config)
        self.__dict__.update(self.init_args)

        sharpness = str2val(string_config, 'sharpn', float, default=sharpness)
        dampening = str2val(string_config, 'dampf', float, default=dampening)
        self.soft_spike = lambda x: dampening * tf.nn.sigmoid(
            sharpness * x) if 'annealing' in string_config else 0

        if 'randompseudod' in string_config:
            spike_functions = [SpikeFunctionGauss, SpikeFunctionCauchy, SpikeFunction, SpikeFunctionSigmoid,
                               ExpSpikeFunction, CappedSkipSpikeFunction, FastSigmoidSpikeFunction]
            spikes = lambda x: [s(x, dampening, sharpness) for s in spike_functions]
            self.random_switch = RandomSwitch([1 / len(spike_functions)] * len(spike_functions))
            self.hard_spike = lambda x: self.random_switch(spikes(x))
        else:
            self.hard_spike = lambda x: ChoosePseudoHeaviside(x, config=string_config, sharpness=sharpness,
                                                              dampening=dampening)

    def build(self, input_shape):
        self.hard_heaviside = self.add_weight(
            name='hard_heaviside', shape=(), initializer=tf.keras.initializers.Constant(1.), trainable=False
        )
        self.built = True

    def call(self, inputs, *args, **kwargs):
        v_sc = inputs
        z = self.hard_heaviside * self.hard_spike(v_sc) + (1 - self.hard_heaviside) * self.soft_spike(v_sc)
        z.set_shape(v_sc.get_shape())
        return z


possible_pseudod = [
    'originalpseudod',
    'exponentialpseudod',
    'gaussianpseudod',
    # 'cauchypseudod',
    'sigmoidalpseudod',
    'fastsigmoidpseudod',
    'cappedskippseudod',
    # 'ntailpseudod',
]


def clean_pseudo_name(pseudod_name):
    pseudod_name = pseudod_name.replace('pseudod', '').replace('original', 'ReLU').replace('sigmoidal',
                                                                                           '$\partial$ sigmoid').replace(
        'fastsigmoid', '$\partial$ fast-sigmoid').replace('cappedskip', 'skip & cap')
    return pseudod_name


def pseudod_color(pseudod_name):
    import matplotlib.pyplot as plt
    i = possible_pseudod.index(pseudod_name)
    cm = plt.get_cmap('tab20b')
    c = cm(.3 + (i - 1) / (len(possible_pseudod) - 1) * .7)
    return c


def draw_pseudods():
    import numpy as np
    import matplotlib as mpl

    # mpl.rcParams['font.family'] = 'serif'
    mpl = load_plot_settings(mpl)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, gridspec_kw={'wspace': .15}, sharey=True, figsize=(20, 5))

    for k in possible_pseudod:
        x = tf.cast(tf.constant(np.linspace(0, 1.5, 1000)), tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = ChoosePseudoHeaviside(x, k + '_sharpn:1')
        grad = tape.gradient(y, x)
        print(k)
        print(np.mean(grad) * 4)

        axs[0].plot(x, grad, color=pseudod_color(k), label=clean_pseudo_name(k))

    exponents = 10 ** np.linspace(-2, 1.2, 7) + 1
    print(exponents)
    for i, k in enumerate(exponents):
        cm = plt.get_cmap('Oranges')
        c = cm(.4 + (i - 1) / (len(exponents) - 1) * .4)
        print(k)
        x = tf.cast(tf.constant(np.linspace(0, 1.5, 1000)), tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = ChoosePseudoHeaviside(x, 'ntailpseudod_tailvalue:' + str(k))
        grad = tape.gradient(y, x)

        axs[1].plot(x, grad, color=c)

    axs[0].set_xlabel('centered voltage')
    axs[0].set_ylabel('surrogate gradient\namplitude')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=pseudod_color(n), lw=4, label=clean_pseudname(n))
                       for n in possible_pseudod]
    axs[0].legend(handles=legend_elements, loc='upper right')
    for ax in axs:
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)
    axs[0].set_xticks([0, 1])
    axs[1].set_xticks([0, 1])
    axs[0].set_yticks([0, 1])
    plot_filename = r'pseudods.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()


def clean_pseudname(name):
    name = name.replace('pseudod', '').replace('original', 'triangular')
    name = name.replace('fastsigmoid', '$\partial$ fast sigmoid')
    name = name.replace('sigmoidal', '$\partial$ sigmoid')
    name = name.replace('cappedskip', 'skip & cap')
    return name


def draw_legend():
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl = load_plot_settings(mpl=mpl)

    legend_elements = [Line2D([0], [0], color=pseudod_color(n), lw=4, label=clean_pseudname(n))
                       for n in possible_pseudod]

    # Create the figure
    fig, ax = plt.subplots(figsize=(3, 3))
    for pos in ['right', 'left', 'bottom', 'top']:
        ax.spines[pos].set_visible(False)
    # ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
    #                 labelleft='off')

    ax.legend(handles=legend_elements, loc='center')

    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                    labelright='off', labelbottom='off')

    plot_filename = r'legend.pdf'
    # fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_pseudods()
    # draw_legend()
