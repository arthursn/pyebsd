import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('savefig', dpi=300, bbox='tight', pad_inches=0)

import pyebsd

if 'fname_local' not in locals() or 'fname_local' not in globals():
    fname_local = ''

reload_data = False
# fname = os.path.join(pyebsd.DIR, 'data', 'QP170-375-30s.ang')
fname = '/home/arthur/Documents/Tohoku2016/EBSD/ang/QP170-375-15_cropped.ang'
# fname = '/home/arthur/Documents/Tohoku2016/EBSD/ang/QP170-375-30s.ang'
# fname = '/home/arthur/Documents/Tohoku2016/EBSD/ang/ADI375-15.ang'
# fname = '/home/arthur/Documents/Tohoku2016/EBSD/ang/Q.ang'
# fname = os.path.join(pyebsd.DIR, 'data', 'dummy.ang')

if (fname != fname_local) or reload_data == True:
    fname_local = fname
    scan = pyebsd.load_scandata(fname_local)

ncols_odd, ncols_even = scan.ncols_odd, scan.ncols_even

R = scan.R
M = scan.M
near = scan.get_neighbors()

C = pyebsd.list_symm()

tr = np.ndarray((scan.N, 6))
tr.fill(-1)

sel = np.ndarray(scan.N, dtype=bool)
sel.fill(True)
sel = (scan.ph == 1) & (scan.CI > .2)

t0 = time.time()
for k in range(6):
    ok = (near[:, k] > 0) & sel & sel[near[:, k]]
    # Matrix multiplication and transposition using einsum notation
    # Equivalent to: np.matmul(M[near[ok,k]], M[ok].transpose([0,2,1]))
    S = np.einsum('ijk,imk->ijm', M[near[ok, k]], M[ok])
    for m in range(len(C)):
        a, b = C[m].nonzero()
        # Trace using Einsum. Equivalent to (S[:,a,b]*C[m,a,b]).sum(axis=1)
        T = np.abs(np.einsum('ij,j->i', S[:, a, b], C[m, a, b]))
        tr[ok, k] = np.max(np.vstack([tr[ok, k], T]), axis=0)
    print(k)
del S, T
print(time.time()-t0)

tr[tr > 3.] = 3.
neg, _ = np.where(tr < 0)
bins = np.arange(-.5, scan.N+.5, 1.)
nneg, _ = np.histogram(neg, bins)

fill = 0
tr[tr < 0] = 4
trmin = np.min(tr, axis=1)
trmin[trmin > 3.5] = fill
tr[tr > 3.5] = 0
travg = np.sum(tr, axis=1)
travg[nneg < 6] = travg[nneg < 6]/(6. - nneg[nneg < 6])
travg[nneg == 6] = fill  # points who don't have any neighbor
trmax = np.max(tr, axis=1)


def tr2ang(tr):
    return np.degrees(np.arccos((tr-1.)/2.))


scan.plot_property(tr2ang(travg), vmax=2, tiling='hex', w=4096)

mis = tr2ang(tr[tr > fill])

fig2, ax2 = plt.subplots()
ax2.hist(mis[mis > 2].ravel(), bins=200, density=True)
ax2.set_xlabel('Misorientation angle (deg)')
ax2.set_ylabel('Frequency')

plt.ion()
plt.show()
