import numpy as np
import pyebsd


def misorientation_between_variants(V):
    mis = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            mis.append(pyebsd.misorientation(V[i], V[j]))
    return mis


def plot_hist_misorientation_variants(V, ax, title=None):
    mis = misorientation_between_variants(V)
    ax.hist(mis, bins=100)
    ax.set_xlabel('Misorientation angle (deg)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)


if __name__ == '__main__':
    from itertools import cycle
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=.3)
    cyaxes = cycle(axes.ravel())

    V = pyebsd.OR()  # KS
    plot_hist_misorientation_variants(V, next(cyaxes), 'KS OR')

    V = pyebsd.OR(ds=([1, 1, 0], [1, 0, 0]))  # NW
    plot_hist_misorientation_variants(V, next(cyaxes), 'NW OR')

    V = pyebsd.OR(ps=([1, 0, 0], [1, 0, 0]), ds=([0, 1, 0], [0, 1, 1]))  # Bain
    plot_hist_misorientation_variants(V, next(cyaxes), 'Bain OR')

    V = pyebsd.OR(ps=([0, 1, 0], [1, 0, 1]), ds=([1, 0, 1], [-1, 1, 1]))  # Pitsch
    plot_hist_misorientation_variants(V, next(cyaxes), 'Pitsch OR')

    plt.show()
