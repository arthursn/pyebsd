import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyebsd


def plot_tests(scan, **kwargs):
    fig, axes = plt.subplots(2, 2)
    scan.plot_property(scan.IQ, cmap='gray', ax=axes[0, 0], **kwargs)
    scan.plot_phase(gray=scan.IQ, ax=axes[0, 1], **kwargs)
    scan.plot_KAM(ax=axes[1, 0], maxmis=60, **kwargs)
    scan.plot_IPF(gray=scan.IQ, ax=axes[1, 1], **kwargs)
    title = 'nrows={}, ncols_odd={}, ncols_even={}, tiling={}'.format(
        scan.nrows, scan.ncols_odd, scan.ncols_even, kwargs.pop('tiling', None))
    fig.suptitle(title, y=1, fontsize=16)
    fig.tight_layout()
    return fig, axes


def drop_first_and_last_columns(data, ncols_odd, ncols_even, odd_row=True):
    ncols = ncols_odd + ncols_even
    if odd_row:
        rm = list(range(0, len(data), ncols))
        rm += list(range(ncols_odd-1, len(data), ncols))
        ncols_odd -= 2
    else:
        rm = list(range(ncols_odd, len(data), ncols))
        rm += list(range(ncols-1, len(data), ncols))
        ncols_even -= 2
    data.drop(rm, inplace=True)
    data.reset_index(inplace=True)
    try:
        data.drop(columns='level_0', inplace=True)
    except:
        pass
    try:
        data.drop(columns='index', inplace=True)
    except:
        pass
    return ncols_odd, ncols_even


def drop_last_row(data, nrows, ncols_odd, ncols_even):
    if nrows & 1 == 1:  # odd nrows
        rm = range(len(data)-ncols_odd, len(data))
    else:
        rm = range(len(data)-ncols_even, len(data))
    data.drop(rm, inplace=True)
    data.reset_index(inplace=True)
    try:
        data.drop(columns='level_0', inplace=True)
    except:
        pass
    try:
        data.drop(columns='index', inplace=True)
    except:
        pass
    return nrows - 1


if __name__ == '__main__':
    fname = os.path.join('..', 'data', 'ADI_bcc_fcc_cropped.ang')
    kwargs = dict(tiling='hex')

    # Test 1: ncols_odd > ncols_even and odd number of rows
    scan1 = pyebsd.load_scandata(fname)
    plot_tests(scan1, **kwargs)

    # Test 2: ncols_odd > ncols_even and even number of rows
    data = scan1.data.copy()
    nrows, ncols_odd, ncols_even = scan1.nrows, scan1.ncols_odd, scan1.ncols_even
    nrows = drop_last_row(data, nrows, ncols_odd, ncols_even)
    scan2 = pyebsd.ScanData(data, 'HexGrid', scan1.dx, scan1.dy, ncols_odd, ncols_even, nrows, scan1.header)
    plot_tests(scan2, **kwargs)

    # Test 3: ncols_odd < ncols_even and odd number of rows
    data = scan1.data.copy()
    nrows, ncols_odd, ncols_even = scan1.nrows, scan1.ncols_odd, scan1.ncols_even
    # drops last column in a odd row twice
    ncols_odd, ncols_even = drop_first_and_last_columns(data, ncols_odd, ncols_even, True)
    scan3 = pyebsd.ScanData(data, 'HexGrid', scan1.dx, scan1.dy, ncols_odd, ncols_even, nrows, scan1.header)
    plot_tests(scan3, **kwargs)

    # Test 4: ncols_odd < ncols_even and even number of rows
    data = scan1.data.copy()
    nrows, ncols_odd, ncols_even = scan1.nrows, scan1.ncols_odd, scan1.ncols_even
    # drops last column in a odd row twice
    ncols_odd, ncols_even = drop_first_and_last_columns(data, ncols_odd, ncols_even, True)
    # drops last row
    nrows = drop_last_row(data, nrows, ncols_odd, ncols_even)
    scan4 = pyebsd.ScanData(data, 'HexGrid', scan1.dx, scan1.dy, ncols_odd, ncols_even, nrows, scan1.header)
    plot_tests(scan4, **kwargs)

    # plt.ion()
    plt.show()
