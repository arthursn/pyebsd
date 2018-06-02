import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

newdir = os.path.join('..', '..')
if newdir not in sys.path:
    sys.path.insert(1, newdir)
import pyebsd


def angle_OR_vars(V, ax=None):
    pairs = list(combinations(list(range(len(V))), 2))
    C = pyebsd.list_symm()
    T = np.dot(C, V).transpose([2, 0, 1, 3])
    mis = [pyebsd.mis(T[p[0]], V[p[1]].T, math='min') for p in pairs]

    return mis


fig, ax = plt.subplots()

V = pyebsd.OR()  # KS
# V = pyebsd.OR(ps=([1,0,0],[1,0,0]), ds=([0,1,0],[0,1,1])) # Bain
# V = pyebsd.OR(ps=([0,1,0],[1,0,1]), ds=([1,0,1],[-1,1,1])) # Pitsch
# V = pyebsd.OR(ds=([1,1,0],[1,0,0]))
# V = pyebsd.OR(ds=([1,1,0],[1,0,0])) # NW
mis = angle_OR_vars(V)
ax.hist(mis, bins=100)


# mis = angle_OR_vars(V, ax=ax)
# ax.hist(mis, bins=100)

# V = pyebsd.OR(ds=([5,12,17],[7,17,17])) # GT
# mis = angle_OR_vars(V, ax=ax)
# ax.hist(mis, bins=100)

plt.show()
