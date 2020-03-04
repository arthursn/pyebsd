import numpy as np
import matplotlib.pyplot as plt


def do_n_rotations(R, p, n):
    ps = [list(p)]
    for k in range(n):
        p = np.dot(R, p)
        ps.append(list(p))
    return ps


if __name__ == '__main__':
    c60 = np.cos(np.pi/3.)
    s60 = np.sin(np.pi/3.)
    R = [[c60, -s60],
         [s60, c60]]

    A = [[1., .5],
         [0., .5*3.**.5]]

    B = [[.5, -.5],
         [0, 1]]

    x_mult6 = []
    y_mult6 = []
    x_mult12 = []
    y_mult12 = []
    neighbors = []
    n = 6
    for ijsum in range(n+1):
        for i in range(1, ijsum+1):
            j = ijsum - i
            if j <= i:
                xyneighbor = []
                jineighbor = []


                p = np.dot(A, [i, j])
                xyneighbor = do_n_rotations(R, p, 5)

                if (j != 0 and i != j):
                    p = np.dot(A, [j, i])
                    # https://codegolf.stackexchange.com/questions/169893/python-shortest-way-to-interleave-items-from-two-lists
                    # Interleave lists
                    xyneighbor = [*sum(zip(xyneighbor, do_n_rotations(R, p, 5)), ())]

                for x, y in xyneighbor:
                    jineighbor.append([(x/c60).round(0).astype(int), (y/s60).round(0).astype(int)])

                print(i, j, ijsum, i**2 + j**2 + i*j, jineighbor)
                # for r in range(1, 6):
                #     xy = np.dot(R60, xy)  # 60 degrees rotation

                neighbors.append(xyneighbor)

    th = np.linspace(0, 2*np.pi, 100)
    cth = np.cos(th)
    sth = np.sin(th)
    for k, xyneighbor in enumerate(neighbors):
        r = (xyneighbor[0][0]**2. + xyneighbor[0][1]**2.)**.5
        pts, = plt.plot(*np.array(xyneighbor).T, marker='.', ls='none', label='{:.3g}'.format(r))
        plt.plot(r*cth, r*sth, color=pts.get_color(), lw=.5)

    plt.gca().set_aspect('equal')
    plt.legend(ncol=2)
    plt.show()
