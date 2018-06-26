import numpy as np
from itertools import permutations


def list_symm(**kwargs):
    """
    List symmetry matrices for cubic symmetry group 
    """
    permt = list(permutations(list(range(3))))
    I = np.diag([1, 1, 1])
    C1 = np.asarray([I[:, arr] for arr in permt])
    C2, C3, C4 = C1.copy(), C1.copy(), C1.copy()
    C2[:, :, 0] = -C1[:, :, 0]
    C3[:, :, 1] = -C1[:, :, 1]
    C4[:, :, 2] = -C1[:, :, 2]
    return np.vstack([C1, C2, C3, C4])


def list_vars(d):
    """
    List all the variants of a family of directions 'd'
    """
    C = list_symm()
    var = set([tuple(v) for v in np.dot(C, d)])
    return np.asarray(list(var))


def stereographic_projection(d, norm=True, coord='cartesian'):
    """
    Returns the coordinates of the stereographic projection of a direction 'd'
    """
    d = np.asarray(d)
    ndim = np.ndim(d)
    shp = np.shape(d)

    if 3 not in shp:
        return

    if ndim == 1:
        d = d.reshape(3, 1)
    elif ndim == 2:
        if shp[0] != 3:
            d = d.transpose([1, 0])
    else:
        return

    if norm:
        d = d/np.linalg.norm(d, axis=0)

    c0, c1 = d[0]/(1.+d[2]), d[1]/(1.+d[2])

    if coord == 'polar':
        r = (c0**2. + c1**2.)**.5
        theta = np.arctan2(c1, c0)
        theta[theta < 0] = theta[theta < 0] + 2*np.pi
        c0, c1 = r, theta

    if ndim == 1:
        c0, c1 = np.float(c0), np.float(c1)

    return c0, c1


def proj2direction(c0, c1):
    d2 = (1-c0**2.-c1**2.)/(1+c0**2.+c1**2.)
    d = [c0*(1.+d2), c1*(1.+d2), d2]
    return d


def mis(A, B, out='deg', math='avg', **kwargs):
    Adim, Bdim = np.ndim(A), np.ndim(B)

    if Bdim > Adim:
        A, B = B, A
        Adim, Bdim = Bdim, Adim
    if (Adim == 3) and (Bdim == 2):
        D = np.tensordot(A, B, axes=[[-1], [-2]])
        x = np.abs(np.trace(D, axis1=1, axis2=2))  # trace
        if out != 'tr':
            x = np.arccos((x-1.)/2.)  # mis x in radians
            if out == 'deg':
                x = np.degrees(x)  # mis x in degrees
        if math == 'avg':
            x = np.mean(x)
        elif math == 'min':
            x = np.min(x)
        elif math == 'max':
            x = np.max(x)
    elif (Adim == 3) and (Bdim == 3):
        if kwargs.pop('vect', True):
            D = np.tensordot(A, B, axes=[[-1], [-2]]).transpose([0, 2, 1, 3])
            x = np.abs(np.trace(D, axis1=2, axis2=3))  # trace
            if out != 'tr':
                x = np.arccos((x-1.)/2.)  # mis x in radians
                if out == 'deg':
                    x = np.degrees(x)  # mis x in degrees
            if math == 'avg':
                x = np.mean(x, axis=0)
            elif math == 'min':
                x = np.min(x, axis=0)
            elif math == 'max':
                x = np.max(x, axis=0)
        else:
            if math is not None:
                x = np.ndarray(len(B))
            else:
                x = np.ndarray((len(B), len(A)))
            for i in range(len(B)):
                D = np.tensordot(A, B[i], axes=[[-1], [-2]])
                y = np.abs(np.trace(D, axis1=1, axis2=2))  # trace
                if out != 'tr':
                    y = np.arccos((y-1.)/2.)  # mis x in radians
                    if out == 'deg':
                        y = np.degrees(y)  # mis x in degrees
                if math == 'avg':
                    x[i] = np.mean(y)
                elif math == 'min':
                    x[i] = np.min(y)
                elif math == 'max':
                    x[i] = np.max(y)
                else:
                    x[i] = y
        del D
    else:
        return
    return x


def reduce_vars(V, trunc=1e-3):
    """
    Returns a reduced number of crystal bases after removing those who are
    equivalent to another already listed.

    V : ndarray shape(N,3,3)
        List of N 3x3 arrays (matrices) representing crystal bases.
    trunc : float
        Maximum misorientation angle (deg) between two bases to consider
        them equivalent to each other.
        Default: 1e-3
    """
    C = list_symm()

    pairs = []
    N = len(V)
    for i in range(N):
        p = []
        U = np.dot(C, V[i])
        for j in range(i, N):
            tr = np.abs(np.trace(np.dot(U, V[j].T), axis1=1, axis2=2))
            mis = np.arccos((tr-1.)/2.)
            if np.degrees(np.min(mis)) < trunc:
                p.append(j)
        pairs.append(p)

    single = [p.pop() for p in pairs if len(p) == 1]
    Vprime = V[single]

    return Vprime
