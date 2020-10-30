import numpy as np
import pyebsd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Symmetry operators sorted according to the KS variants convention
    C = pyebsd.list_cubic_symmetry_operators_KS()
    KS = pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[-1, 0, 1], [-1, -1, 1]],
                   single=True)  # only the first matrix

    # The orientations in M_KS are in the order of the variants convention
    # published in papers such as:
    # T. Furuhara, T. Maki, Mater. Sci. Eng. A 312 (2001) 145–154.
    M_KS = np.matmul(KS, C.transpose([0, 2, 1]))  # Mj = M0 . C^-1

    ax = pyebsd.plot_PF(M_KS[0], marker='$1$', ms=8, verbose=False)
    for i, M in enumerate(M_KS[1:]):
        # Plot 001 pole figures fro each variant
        pyebsd.plot_PF(M, marker='${}$'.format(i+2), ms=8, ax=ax, verbose=False)

    ax.text(1.05, 0, r'[100]$\gamma$', size=15)
    ax.text(0, 1.05, r'[010]$\gamma$', ha='center', size=15)

    # Calculate misorientation with respect to the first variant
    variant = 0
    misang = pyebsd.misorientation(M=M_KS,
                                   neighbors=np.full(shape=[len(M_KS), 1],
                                                     fill_value=variant),
                                   verbose=False)

    print('Misorientation angle with respect to variant 1')
    for v, mis in enumerate(misang):
        print('{:2d}: {:g}'.format(v, float(mis)))

    plt.show()
