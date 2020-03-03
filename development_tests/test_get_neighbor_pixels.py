import numpy as np
import matplotlib.pyplot as plt


def do_n_rotations(R, p, n):
    ps = [list(p)]
    for k in range(n):
        p = np.dot(R, p)
        ps.append(list(p))
    return ps


if __name__ == '__main__':
    nturns = 6
    theta = 2.*np.pi/nturns
    R = [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    A = [[1., .5],
         [0., .5*3.**.5]]

    x_mult6 = []
    y_mult6 = []
    x_mult12 = []
    y_mult12 = []
    neighbors = []
    n = 10
    for ijsum in range(n+1):
        for i in range(1, ijsum+1):
            j = ijsum - i
            if j <= i:
                ijneighbor = []

                print(i, j, ijsum, i**2 + j**2 + i*j)

                p = np.dot(A, [i, j])
                ijneighbor = do_n_rotations(R, p, 5)

                if (j != 0 and i != j):
                    p = np.dot(A, [j, i])
                    # https://codegolf.stackexchange.com/questions/169893/python-shortest-way-to-interleave-items-from-two-lists
                    # Interleave lists
                    ijneighbor = [*sum(zip(ijneighbor, do_n_rotations(R, p, 5)), ())]

                neighbors.append(ijneighbor)

    for k, ijneighbor in enumerate(neighbors):
        plt.plot(*np.array(ijneighbor).T, marker='.', lw=.5, label=str(k))

    plt.gca().set_aspect('equal')
    plt.legend(ncol=2)
    plt.show()
