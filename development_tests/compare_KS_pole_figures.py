import numpy as np
import matplotlib.pyplot as plt
import pyebsd

if __name__ == '__main__':
    # T. Furuhara, T. Maki, Mater. Sci. Eng. A 312 (2001) 145–154.
    T_KS = []
    T_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[-1, 0, 1], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[-1, 0, 1], [-1, 1, -1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[0, 1, -1], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[0, 1, -1], [-1, 1, -1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[1, -1, 0], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[1, -1, 0], [-1, 1, -1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[1, 0, -1], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[1, 0, -1], [-1, 1, -1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[-1, -1, 0], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[-1, -1, 0], [-1, 1, -1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[0, 1, 1], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, -1, 1], [0, 1, 1]], ds=[[0, 1, 1], [-1, 1, -1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[0, -1, 1], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[0, -1, 1], [-1, 1, -1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[-1, 0, -1], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[-1, 0, -1], [-1, 1, -1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[1, 1, 0], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[-1, 1, 1], [0, 1, 1]], ds=[[1, 1, 0], [-1, 1, -1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[-1, 1, 0], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[-1, 1, 0], [-1, 1, -1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[0, -1, -1], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[0, -1, -1], [-1, 1, -1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[1, 0, 1], [-1, -1, 1]], single=True))
    T_KS.append(pyebsd.OR(ps=[[1, 1, -1], [0, 1, 1]], ds=[[1, 0, 1], [-1, 1, -1]], single=True))

    ax = pyebsd.plot_PF(T_KS[0], marker='$1$', ms=8, verbose=False)
    for i, T in enumerate(T_KS[1:]):
        # Plot 001 pole figures fro each variant
        pyebsd.plot_PF(T, marker='${}$'.format(i+2), ms=8, ax=ax, verbose=False)
    ax.set_title("Orientation matrices calculated from Furuhara's paper")

    T_KS_pyebsd = pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[-1, 0, 1], [-1, -1, 1]], single=True)
    C_KS = pyebsd.list_cubic_symmetry_operators_KS()
    T_KS_pyebsd = np.matmul(T_KS_pyebsd, C_KS.transpose(0, 2, 1))
    ax = pyebsd.plot_PF(T_KS_pyebsd[0], marker='$1$', ms=8, verbose=False)
    for i, T in enumerate(T_KS_pyebsd[1:]):
        # Plot 001 pole figures fro each variant
        pyebsd.plot_PF(T, marker='${}$'.format(i+2), ms=8, ax=ax, verbose=False)
    ax.set_title("Orientation matrices calculated using symmetry operators from pyebsd")

    plt.show()
