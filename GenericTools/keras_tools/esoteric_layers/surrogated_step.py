import tensorflow as tf
import math

from GenericTools.stay_organized.utils import str2val


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
        dz_dv_scaled = dampening * 1 / (1 + xabs) ** 2
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(sharpness)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def NTailSpikeFunction(v_scaled, dampening, sharpness, tail):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        xabs = tf.abs(v_scaled * sharpness)
        factor = (tail - 1) / 2
        dz_dv_scaled = dampening * 1 / (1 + xabs / factor) ** tail
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(sharpness), tf.zeros_like(tail)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def RectangularSpikeFunction(v_scaled, dampening, sharpness):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        cap = dampening * tf.cast(tf.less(tf.abs(2 * sharpness * v_scaled), 1), dtype=tf.float32)
        return [dy * cap, tf.zeros_like(dampening), tf.zeros_like(sharpness)]

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
def SpikeFunctionMultiGaussian(v_scaled, dampening, sharpness, h, sl, sr, hl, hr, shiftl, shiftr):
    # Accurate and efficient time-domain classification
    # with adaptive spiking recurrent neural networks
    # Bojian Yin, Federico Corradi and Sander M. Bohté

    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.keras.backend.floatx())

    def grad(dy):  # best m found 25
        x = v_scaled * sharpness
        # h = .15
        # s = 2
        A = (1 + h - 2 * h * tf.exp(-1 / (sl + sr ** 2))) ** (-1)  # 1
        width = (A * (1 + h) * tf.sqrt(2 * math.pi) - A * h * (sl + sr) * tf.sqrt(2 * math.pi)) ** (-1)  # .5

        central_g = (1 + h) * tf.exp(-tf.pow(x, 2) / (2 * width ** 2))
        left_g = hl * tf.exp(-tf.pow(x - shiftl * width, 2) / (2 * sl ** 2 * width ** 2))
        right_g = hr * tf.exp(-tf.pow(x + shiftr * width, 2) / (2 * sr ** 2 * width ** 2))
        dz_dv_scaled = central_g - left_g - right_g
        dz_dv_scaled = dampening * A * dz_dv_scaled
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(sharpness)] + [tf.zeros_like(h)] * 7

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
        dz_dv_scaled = dampening * tf.cast(tf.math.equal(v_scaled, 0), tf.keras.backend.floatx())
        # dz_dv_scaled = 0
        return [dy * dz_dv_scaled, tf.zeros_like(dampening)]

    return tf.identity(z_, name="SpikeFunctionDeltaDirac"), grad


@tf.custom_gradient
def SpikeFunctionExpsPowerLaws(v_scaled, dampening, sharpness, a, b, c, d, e, f, g, h, i, l, m):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.keras.backend.floatx())

    def grad(dy):  # 20 best multiplicative factors found
        x = v_scaled * sharpness + m
        dz_dv_scaled = (a * tf.exp(b * x) + e * 1 / (1 + f * tf.abs(x) ** (1 + i))) \
                       * tf.cast(tf.greater(-x, 0.), dtype=tf.keras.backend.floatx()) \
                       + (c * tf.exp(- d * x) + h * 1 / (1 + g * tf.abs(x) ** (1 + l))) \
                       * tf.cast(tf.greater(x, 0.), dtype=tf.keras.backend.floatx())
        # dz_dv_scaled = 1 / (1 + tf.pow(math.pi * v_scaled * sharpness, 2)) * dampening
        dz_dv_scaled = dampening * dz_dv_scaled / tf.math.maximum(a + e, c + h)
        return [dy * dz_dv_scaled] + [tf.zeros_like(a)] * 13

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.custom_gradient
def MNTailSpikeFunction(v_scaled, dampening, sharpness, tail1, tail2, tail3, c1, c2, c3, h1, h2, h3):
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        x1 = tf.abs((v_scaled - c1) * sharpness)
        peak1 = h1 / (1 + 2 * x1 / (tail1 - 1)) ** tail1
        x2 = tf.abs((v_scaled - c2) * sharpness)
        peak2 = h2 / (1 + 2 * x2 / (tail2 - 1)) ** tail2
        x3 = tf.abs((v_scaled - c3) * sharpness)
        peak3 = h3 / (1 + 2 * x3 / (tail3 - 1)) ** tail1

        peaks = peak1 + peak2 + peak3
        dz_dv_scaled = dampening * peaks
        return [dy * dz_dv_scaled, tf.zeros_like(dampening), tf.zeros_like(sharpness)] + [tf.zeros_like(tail1)] * 9

    return tf.identity(z_, name="SpikeFunction"), grad


def ChoosePseudoHeaviside(v_sc, config='', sharpness=1, dampening=1):
    sharpness = str2val(config, 'sharpn', float, default=sharpness)
    dampening = str2val(config, 'dampf', float, default=dampening)

    if 'gaussianpseudod' in config:
        z = SpikeFunctionGauss(v_sc, dampening, sharpness)

    elif 'cauchypseudod' in config:
        z = SpikeFunctionCauchy(v_sc, dampening, sharpness)

    elif 'originalpseudod' in config or 'triangularpseudod' in config:
        z = SpikeFunction(v_sc, dampening, sharpness)

    elif 'sigmoidalpseudod' in config:
        z = SpikeFunctionSigmoid(v_sc, dampening, sharpness)

    elif 'exponentialpseudod' in config:
        z = ExpSpikeFunction(v_sc, dampening, sharpness)

    elif 'cappedskippseudod' in config or 'rectangularpseudod' in config:
        z = RectangularSpikeFunction(v_sc, dampening, sharpness)

    elif 'fastsigmoidpseudod' in config:
        z = FastSigmoidSpikeFunction(v_sc, dampening, sharpness)

    elif 'ntailpseudod' in config:
        tail = str2val(config, 'tailvalue', float, default=1.56)
        z = NTailSpikeFunction(v_sc, dampening, sharpness, tail)

    elif 'mgausspseudod' in config:
        # tail = str2val(config, 'tailvalue', float, default=1.1)
        z = SpikeFunctionMultiGaussian(v_sc, dampening, sharpness, .15, 2, 2, .15, .15, 1, 1)

    elif 'reluspike' in config:
        z = dampening * tf.nn.relu(sharpness * v_sc)

    elif 'sigspike' in config:
        z = dampening * tf.nn.sigmoid(sharpness * v_sc)

    elif 'geluspike' in config:
        z = dampening * tf.nn.gelu(sharpness * v_sc)

    elif 'softplusspike' in config:
        z = dampening * tf.math.softplus(sharpness * v_sc)

    elif 'eppseudod' in config:
        params = []
        defaults = [-0.016451614574342632, 1.7112170716428443, -1.094737676894167, 0.93718267274918,
                    0.054550157179523155, 0.10135086310595437, 0.9907439317066982, 1.071867159728163,
                    -0.32863050315427816, 1.0317211556163626, 0.2]
        stacki = str2val(config, 'stacki', int, default=-1)
        stacki = '' if stacki == -1 else stacki

        for s, d in zip(
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm'],
                defaults):
            param = str2val(config, f'{s}{stacki}', float, default=d, exact_flag=True)
            if not s == 'm':
                param = abs(param)
            params.append(param)

        z = SpikeFunctionExpsPowerLaws(v_sc, dampening, sharpness, *params)
    else:
        z = FastSigmoidSpikeFunction(v_sc, dampening, sharpness)

    return z


def OneLearnableNTail(self, n_in):
    self.sharpness = self.hard_heaviside = self.add_weight(
        name='sharp', shape=(n_in,), initializer=tf.keras.initializers.Constant(1.), trainable=True
    )
    self.dampening = self.hard_heaviside = self.add_weight(
        name='damp', shape=(n_in,), initializer=tf.keras.initializers.Constant(1.), trainable=True
    )
    self.tail = self.hard_heaviside = self.add_weight(
        name='tail', shape=(n_in,), initializer=tf.keras.initializers.Constant(2.), trainable=True
    )
    self.hard_spike = lambda x: NTailSpikeFunction(x, self.dampening, self.sharpness, self.tail)
    return self


def MLearnableGauss(self, n_in):
    self.sharpness = self.hard_heaviside = self.add_weight(
        name='sharp', shape=(n_in,), initializer=tf.keras.initializers.Constant(1.), trainable=True
    )
    self.dampening = self.hard_heaviside = self.add_weight(
        name='damp', shape=(n_in,), initializer=tf.keras.initializers.Constant(1.), trainable=True
    )
    self.h = self.hard_heaviside = self.add_weight(
        name='h', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(-0.05, 1.), trainable=True
    )
    self.sl = self.hard_heaviside = self.add_weight(
        name='sl', shape=(n_in,), initializer=tf.keras.initializers.Constant(2.), trainable=True
    )
    self.sr = self.hard_heaviside = self.add_weight(
        name='sr', shape=(n_in,), initializer=tf.keras.initializers.Constant(2.), trainable=True
    )
    self.hl = self.hard_heaviside = self.add_weight(
        name='hl', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(-0.05, 1.), trainable=True
    )
    self.hr = self.hard_heaviside = self.add_weight(
        name='hr', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(-0.05, 1.), trainable=True
    )
    self.shiftl = self.hard_heaviside = self.add_weight(
        name='shiftl', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(0.05, 1.5), trainable=True
    )
    self.shiftr = self.hard_heaviside = self.add_weight(
        name='shiftr', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(0.05, 1.5), trainable=True
    )
    self.hard_spike = lambda x: SpikeFunctionMultiGaussian(x, self.dampening, self.sharpness, self.h, self.sl,
                                                           self.sr, self.hl, self.hr, self.shiftl,
                                                           self.shiftr)
    return self


def MLearnableTails(self, n_in):
    self.sharpness = self.hard_heaviside = self.add_weight(
        name='sharp', shape=(n_in,), initializer=tf.keras.initializers.Constant(1.), trainable=True
    )
    self.dampening = self.hard_heaviside = self.add_weight(
        name='damp', shape=(n_in,), initializer=tf.keras.initializers.Constant(1.), trainable=True
    )
    self.tail1 = self.hard_heaviside = self.add_weight(
        name='tail1', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(-0.05, 1.), trainable=True
    )
    self.tail2 = self.hard_heaviside = self.add_weight(
        name='tail2', shape=(n_in,), initializer=tf.keras.initializers.Constant(2.), trainable=True
    )
    self.tail3 = self.hard_heaviside = self.add_weight(
        name='tail3', shape=(n_in,), initializer=tf.keras.initializers.Constant(2.), trainable=True
    )
    self.c1 = self.hard_heaviside = self.add_weight(
        name='c1', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(-0.05, 1.), trainable=True
    )
    self.c2 = self.hard_heaviside = self.add_weight(
        name='c2', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(-0.05, 1.), trainable=True
    )
    self.c3 = self.hard_heaviside = self.add_weight(
        name='c3', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(0.05, 1.5), trainable=True
    )
    self.h1 = self.hard_heaviside = self.add_weight(
        name='h1', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(0.05, 1.5), trainable=True
    )
    self.h2 = self.hard_heaviside = self.add_weight(
        name='h2', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(0.05, 1.5), trainable=True
    )
    self.h3 = self.hard_heaviside = self.add_weight(
        name='h3', shape=(n_in,), initializer=tf.keras.initializers.RandomUniform(0.05, 1.5), trainable=True
    )
    self.hard_spike = lambda x: MNTailSpikeFunction(x, self.dampening, self.sharpness, self.tail1, self.tail2,
                                                    self.tail3, self.c1, self.c2, self.c3, self.h1, self.h2, self.h3)
    return self


class SurrogatedStep(tf.keras.layers.Layer):
    """
    Layer for a Heaviside with a surrogate gradient for the backward pass
    """

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, dampening=1., sharpness=1., config='', **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(config=config)
        self.__dict__.update(self.init_args)

        sharpness = str2val(config, 'sharpn', float, default=sharpness)
        dampening = str2val(config, 'dampf', float, default=dampening)
        self.soft_spike = lambda x: \
            dampening * tf.nn.sigmoid(sharpness * x) if 'annealing' in config else 0

        self.hard_spike = lambda x: ChoosePseudoHeaviside(x, config=config, sharpness=sharpness,
                                                          dampening=dampening)

    def build(self, input_shape):
        n_in = input_shape[-1]
        self.hard_heaviside = self.add_weight(
            name='hard_heaviside', shape=(), initializer=tf.keras.initializers.Constant(1.), trainable=False
        )
        self.built = True

        if 'learnablepseudod' in self.config:

            if 'mgauss' in self.config:
                print('here')
                self = MLearnableGauss(self, n_in)
            elif 'mtail' in self.config:
                self = MLearnableTails(self, n_in)
            else:
                self = OneLearnableNTail(self, n_in)

    def call(self, inputs, *args, **kwargs):
        v_sc = inputs
        z = self.hard_heaviside * self.hard_spike(v_sc) + (1 - self.hard_heaviside) * self.soft_spike(v_sc)
        z.set_shape(v_sc.get_shape())
        return z


possible_pseudod = [
    'originalpseudod',
    'exponentialpseudod',
    'gaussianpseudod',
    'sigmoidalpseudod',
    'fastsigmoidpseudod',
    'cappedskippseudod',
    # 'eppseudod'
    # 'ntailpseudod',
    # 'mgausspseudod',
]


def clean_pseudo_name(pseudod_name):
    pseudod_name = pseudod_name.replace('pseudod', '') \
        .replace('original', 'ReLU').replace('sigmoidal', '$\partial$ sigmoid') \
        .replace('fastsigmoid', '$\partial$ fast-sigmoid').replace('cappedskip', 'skip & cap')
    return pseudod_name


def pseudod_color(pseudod_name):
    import matplotlib.pyplot as plt
    # 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r'
    if not pseudod_name == 'ntailpseudod':
        i = possible_pseudod.index(pseudod_name)
        if pseudod_name == 'fastsigmoidpseudod':
            i = 4.2
        elif pseudod_name == 'cappedskippseudod':
            i = 5.  # 5.5

        cm = plt.get_cmap('tab20b')  # tab20b
        c = cm(.3 + (i - 1) / (len(possible_pseudod) - 1) * .7)
    else:
        cm = plt.get_cmap('Oranges')
        c = cm(.5)
    return c


def draw_pseudods():
    import numpy as np
    import matplotlib as mpl
    from GenericTools.stay_organized.mpl_tools import load_plot_settings

    # mpl.rcParams['font.family'] = 'serif'
    mpl = load_plot_settings(mpl)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, gridspec_kw={'wspace': .1}, sharey=False, figsize=(10, 5))

    for k in possible_pseudod:
        # x = tf.cast(tf.constant(np.linspace(0, 1.5, 1000)), tf.float32)
        x = tf.cast(tf.constant(np.linspace(-1.5, 1.5, 1000)), tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = ChoosePseudoHeaviside(x, k + '_sharpn:1')
        grad = tape.gradient(y, x)
        print(k)
        print(np.mean(grad) * 4)

        c = pseudod_color(k)
        print(c)
        cint = (int(255 * i) for i in c)
        print(cint)
        print(k, '#{:02x}{:02x}{:02x}'.format(*cint))
        axs[0].plot(x, grad, color=c, label=clean_pseudo_name(k))

    n_exps = 7
    exponents = 10 ** np.linspace(-2, 1.2, n_exps) + 1

    cm = plt.get_cmap('Oranges')
    for i, k in enumerate(exponents):
        c = cm(.4 + i / (len(exponents) - 1) * .4)
        x = tf.cast(tf.constant(np.linspace(0, 1.5, 1000)), tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = ChoosePseudoHeaviside(x, 'ntailpseudod_tailvalue:' + str(k))
        grad = tape.gradient(y, x)

        print(c)
        cint = (int(255 * i) for i in c)
        print(cint)
        print(k, '#{:02x}{:02x}{:02x}'.format(*cint))
        axs[1].plot(x, grad, color=c)

    n_grad = 100
    gradient = np.linspace(.4, .8, n_grad)[::-1]
    gradient = np.vstack((gradient, gradient))

    ax = fig.add_axes([.90, 0.2, .015, .6])
    ax.imshow(gradient.T, aspect='auto', cmap=cm)
    ax.text(-0.01, 0.5, '$q$-PseudoSpike \t', va='center', ha='right', fontsize=16, transform=ax.transAxes)
    ax.text(8, 0.5, '$q$ \t', va='center', ha='right', fontsize=16, transform=ax.transAxes)
    ax.set_yticks([])
    ax.set_xticks([])

    loc = [-.5 + i / (n_exps - 1) * n_grad for i in range(n_exps)]
    exponents = [round(e, 2) for e in 10 ** np.linspace(-2, 1.2, n_exps) + 1][::-1]

    ax.set_yticks(loc)
    ax.set_yticklabels(exponents)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    for pos in ['right', 'left', 'bottom', 'top']:
        ax.spines[pos].set_visible(False)

    # axs[0].set_xlabel('centered voltage')
    axs[0].set_ylabel('surrogate gradient\namplitude')
    axs[1].set_xlabel('centered voltage')
    axs[1].set_ylabel('surrogate gradient\namplitude')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=pseudod_color(n), lw=4, label=clean_pseudname(n))
                       for n in possible_pseudod]
    axs[0].legend(handles=legend_elements, loc='best', bbox_to_anchor=(0.4, 0.5, 0.4, 0.5))
    for ax in axs:
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)
    axs[0].set_xticks([0, 1])
    axs[1].set_xticks([0, 1])
    axs[0].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_yticks([], minor=True)
    # axs[0].set_yticks([0, 1])
    axs[0].set_yticks([0, 2])
    axs[1].set_yticks([0, 1])

    plot_filename = r'pseudods.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()


def clean_pseudname(name):
    name = name.replace('pseudod', '').replace('original', 'triangular')
    name = name.replace('fastsigmoid', '$\partial$ fast sigmoid')
    name = name.replace('sigmoidal', '$\partial$ sigmoid')
    name = name.replace('cappedskip', 'rectangular')
    name = name.replace('ntail', '$q$-PseudoSpike')
    return name


def draw_legend():
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from GenericTools.stay_organized.mpl_tools import load_plot_settings
    mpl = load_plot_settings(mpl=mpl)

    legend_elements = [Line2D([0], [0], color=pseudod_color(n), lw=4, label=clean_pseudname(n))
                       for n in possible_pseudod]

    # Create the figure
    fig, ax = plt.subplots(figsize=(3.5, .5))
    for pos in ['right', 'left', 'bottom', 'top']:
        ax.spines[pos].set_visible(False)
    # ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
    #                 labelleft='off')

    ax.legend(ncol=7, handles=legend_elements, loc='center')

    ax.axis('off')
    ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                   labelright='off', labelbottom='off')
    # plt.tight_layout(pad=0)
    plt.tight_layout(pad=0., w_pad=0., h_pad=0.)

    plot_filename = r'legend.pdf'
    # fig.tight_layout(pad=0)
    fig.savefig(plot_filename, bbox_inches='tight', pad_inches=0)
    # fig.savefig(plot_filename)
    plt.show()


def draw_legend_mini():
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from GenericTools.stay_organized.mpl_tools import load_plot_settings
    mpl = load_plot_settings(mpl=mpl)

    legend_elements = [
        Line2D([0], [0], color='w', lw=4, label='4 seeds'),
        Line2D([0], [0], color=pseudod_color('originalpseudod'), lw=4, label='mean'),
        Line2D([0], [0], color=pseudod_color('originalpseudod'), lw=16, label='std', alpha=0.5),
    ]

    # Create the figure
    fig, ax = plt.subplots(figsize=(3, 3))
    for pos in ['right', 'left', 'bottom', 'top']:
        ax.spines[pos].set_visible(False)
    # ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
    #                 labelleft='off')

    ax.legend(handles=legend_elements, loc='center', frameon=False)

    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
                    labelright='off', labelbottom='off')

    plot_filename = r'meanstd.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # draw_pseudods()
    draw_legend()
    # draw_legend_mini()
