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
    T_KS = np.array(T_KS)

    C = pyebsd.list_cubic_symmetry_operators_KS()

    C_KS = []

    for j in range(len(C)):
        Cj = np.matmul(T_KS[j].T, np.matmul(C, T_KS[0]))
        cnt_ones = np.count_nonzero(np.abs(Cj).round(6) == 1, axis=1).sum(axis=1)
        i, = np.where(cnt_ones == 3)

        if len(i) == 1:
            i = i[0]
        else:
            raise Exception('Exactly one match was expected')

        print(i, j)
        Cj = T_KS[j].T.dot(C[i]).dot(T_KS[0])
        C_KS.append(Cj.round(6).astype(int))

    C_KS = np.array(C_KS)

    # ax = pyebsd.plot_PF(T_KS[0], marker='$1$', ms=8, verbose=False)
    # for i, T in enumerate(T_KS[1:]):
    #     # Plot 001 pole figures fro each variant
    #     pyebsd.plot_PF(T, marker='${}$'.format(i+2), ms=8, ax=ax, verbose=False)
    # ax.set_title("Orientation matrices calculated from Furuhara's paper")

    # T_KS_new = np.matmul(T_KS[0], C_KS.transpose(0, 2, 1))
    # ax = pyebsd.plot_PF(T_KS_new[0], marker='$1$', ms=8, verbose=False)
    # for i, T in enumerate(T_KS_new[1:]):
    #     # Plot 001 pole figures fro each variant
    #     pyebsd.plot_PF(T, marker='${}$'.format(i+2), ms=8, ax=ax, verbose=False)
    # ax.set_title("Orientation matrices calculated using symmetry operators from pyebsd")

    # plt.show()
