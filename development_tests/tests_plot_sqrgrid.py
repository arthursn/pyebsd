import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyebsd


def plot_tests(scan, **kwargs):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    scan.plot_property(scan.IQ, cmap='gray', ax=axes[0, 0], **kwargs)
    scan.plot_phase(gray=scan.IQ, ax=axes[0, 1], **kwargs)
    scan.plot_KAM(distance=1, ax=axes[1, 0], **kwargs)
    scan.plot_IPF(gray=scan.IQ, ax=axes[1, 1], **kwargs)
    title = 'nrows={}, ncols={}'.format(scan.nrows, scan.ncols)
    fig.suptitle(title, y=1, fontsize=16)
    fig.tight_layout()
    return fig, axes


if __name__ == '__main__':
    fname = os.path.join('..', 'data', 'bcc_sqrgrid.ang')
    kwargs = dict()

    scan = pyebsd.load_scandata(fname)
    plot_tests(scan, **kwargs)

    # plt.ion()
    plt.show()
