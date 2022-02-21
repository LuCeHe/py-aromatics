import numpy as np
import matplotlib.pyplot as plt


# original: https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, den)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, den)

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


def load_plot_settings(mpl=None, pd=None, figsize=(10, 10)):
    output = ()
    if not mpl is None:
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['xtick.major.pad'] = '4'
        # print(mpl.rcParams.keys())
        mpl.rcParams['xtick.major.size'] = '2'

        large = 22
        med = 16
        small = 12  # 12

        params = {'axes.titlesize': large,
                  'legend.fontsize': med,
                  'figure.figsize': figsize,
                  'axes.labelsize': med,
                  'xtick.labelsize': small,
                  'ytick.labelsize': small,
                  'figure.titlesize': large}
        mpl.rcParams.update(params)
        output += (mpl,)

    if not pd is None:
        pd.set_option('display.max_columns', None)
        pd.set_option('max_colwidth', 1)
        pd.set_option('precision', 3)
        pd.options.display.width = 500
        # pd.set_option('display.float_format', lambda x: '%.6f' % x)
        # pd.options.display.max_colwidth = 16
        output += (pd,)

    output = output[0] if len(output) == 1 else output
    return output
