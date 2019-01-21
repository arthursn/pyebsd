def remove_redundant_variants(V, trunc=1e-3):
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
    C = list_cubic_symmetry_operators()

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

    return Vprime, pairs
