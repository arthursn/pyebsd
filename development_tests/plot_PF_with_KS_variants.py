import numpy as np
import pyebsd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Symmetry operators sorted according to the KS variants convention
    C = pyebsd.list_cubic_symmetry_operators_KS()
    KS = pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[-1, 0, 1], [-1, -1, 1]],
                   single=True)  # only the first matrix

    # The orientations in M_KS are in the order of the variants convetion
    # published in papers such as:
    # T. Furuhara, T. Chiba, T. Kaneshita, H. Wu, G. Miyamoto, Metall.
    # Mater. Trans. A 48 (2017) 2739–2752.
    M_KS = np.tensordot(KS, C, axes=[[-1], [-2]]).transpose([1, 0, 2])

    ax = pyebsd.plot_PF(M_KS[0], marker='$1$', ms=8, verbose=False)
    for i, M in enumerate(M_KS[1:]):
        # Plot 001 pole figures fro each variant
        pyebsd.plot_PF(M, marker='${}$'.format(i+2), ms=8, ax=ax, verbose=False)

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
