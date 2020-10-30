import numpy as np
import matplotlib.pyplot as plt
import pyebsd

if __name__ == '__main__':
    # T. Furuhara, T. Maki, Mater. Sci. Eng. A 312 (2001) 145–154.
    M_KS = []
    M_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[-1, 0, 1], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[-1, 0, 1], [-1, 1, -1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[0, 1, -1], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[0, 1, -1], [-1, 1, -1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[1, -1, 0], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[1, -1, 0], [-1, 1, -1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[1, 0, -1], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[1, 0, -1], [-1, 1, -1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[-1, -1, 0], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[-1, -1, 0], [-1, 1, -1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[0, 1, 1], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[0, 1, 1], [-1, 1, -1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[0, -1, 1], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[0, -1, 1], [-1, 1, -1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[-1, 0, -1], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[-1, 0, -1], [-1, 1, -1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[1, 1, 0], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[1, 1, 0], [-1, 1, -1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[-1, 1, 0], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[-1, 1, 0], [-1, 1, -1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[0, -1, -1], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[0, -1, -1], [-1, 1, -1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[1, 0, 1], [-1, -1, 1]], single=True))
    M_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[1, 0, 1], [-1, 1, -1]], single=True))

    ax = pyebsd.plot_PF(M_KS[0], marker='$1$', ms=8, verbose=False)
    for i, M in enumerate(M_KS[1:]):
        # Plot 001 pole figures fro each variant
        pyebsd.plot_PF(M, marker='${}$'.format(i+2), ms=8, ax=ax, verbose=False)
    ax.set_title("Orientation matrices calculated from Furuhara's paper")

    M_KS_pyebsd = pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[-1, 0, 1], [-1, -1, 1]], single=True)
    C_KS = pyebsd.list_cubic_symmetry_operators_KS()
    M_KS_pyebsd = np.matmul(M_KS_pyebsd, C_KS.transpose(0, 2, 1))
    ax = pyebsd.plot_PF(M_KS_pyebsd[0], marker='$1$', ms=8, verbose=False)
    for i, M in enumerate(M_KS_pyebsd[1:]):
        # Plot 001 pole figures fro each variant
        pyebsd.plot_PF(M, marker='${}$'.format(i+2), ms=8, ax=ax, verbose=False)
    ax.set_title("Orientation matrices calculated using symmetry operators from pyebsd")

    plt.show()
