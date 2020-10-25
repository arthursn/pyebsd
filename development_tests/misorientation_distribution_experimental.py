import os
import numpy as np
import matplotlib.pyplot as plt
import pyebsd


def misorientation_between_variants(V):
    C = pyebsd.list_cubic_symmetry_operators()
    mis = []
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            mis.append(pyebsd.misorientation_two_rotations(np.dot(C, V[i]), V[j].T, math='min'))
    return mis


if __name__ == '__main__':
    fname = os.path.join('..', 'data', 'ADI_bcc_fcc.ang')
    scan = pyebsd.load_scandata(fname)

    sel = (scan.ph == 1) & (scan.CI > .2)
    # Experimental misrorientations
    mis_exp = pyebsd.misorientation(scan.M, scan.get_neighbors(1, sel=sel), sel).ravel()
    # Expected misorientations between KS variants
    mis_calc = misorientation_between_variants(pyebsd.OR())  # for KS OR variants

    fig, ax = plt.subplots()

    ax.hist(mis_exp[mis_exp > 2], bins=200, density=True, label='Experimental')
    ax.hist(mis_calc, bins=500, density=True, label='Theoretical')
    ax.legend()
    ax.set_xlabel('Misorientation angle (deg)')
    ax.set_ylabel('Frequency')

    plt.show()
