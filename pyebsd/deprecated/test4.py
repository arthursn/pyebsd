#!/usr/bin/python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

DIRS = ['/home/arthur/Dropbox/python', 'Z:\\Arthur', 'D:']
for DIR in DIRS:
    if DIR not in sys.path:
        sys.path.insert(1, DIR)
import pyebsd
from pyebsd import uvw_label

if 'fname_local' not in locals() or 'fname_local' not in globals():
    fname_local = ''
    
# fname = os.path.join(pyebsd.DIR, 'data', 'QP170-375-30s.ang')
fname = '/home/arthur/Documents/Tohoku2016/EBSD/ang/QP170-375-15_cropped.ang'
# fname = os.path.join(pyebsd.DIR, 'data', 'dummy.ang')

# if 'data' not in locals() or 'data' not in globals() or reload_data is True:
if fname != fname_local:
    fname_local = fname
    scan = pyebsd.Scandata(fname_local)

    # data, grid, dx, dy, ncols_odd, ncols_even, nrows = pyebsd.load_ang_file(fname_local)
    # x, y, IQ, CI, ph = data[:,3], data[:,4], data[:,5], data[:,6], data[:,7]
    # phi1, Phi, phi2 = data[:,0], data[:,1], data[:,2]
    # R = pyebsd.euler_rotation(phi1, Phi, phi2)

# plot IPF
# ax, img = scan.plot_IPF(sel=(scan.ph==2) & (scan.CI>.2))

ncols_odd, ncols_even = scan.ncols_odd, scan.ncols_even
# nrows, ncols = ph.shape


# neigh = np.ndarray(ph.shape, dtype=object)

i, j = scan.i, scan.j
# ind0 = scan.ij2ind(i, j)
R = scan.R
M = scan.M

C = pyebsd.list_cubic_symmetry_operators()
# print i, j, ind

i_near = np.ndarray(6, dtype=int)
j_near = np.ndarray(6, dtype=int)
near = np.ndarray(6, dtype=int)

n_near = np.ndarray(i.shape)
a = np.ndarray(scan.N, dtype=float)
b = np.ndarray(scan.N, dtype=object)

sel = scan.ph == 2

import time
t0 = time.time()
N = np.count_nonzero(sel)
each = N/50

n_array = np.ndarray(N, dtype=object)
for k in scan.ind[sel]:
    i0, j0 = i[k], j[k]
    i1_, i1 = i0-1, i0+1
    j2_, j1_, j1, j2 = j0-2, j0-1, j0+1, j0+2

    j_near[:] = [j2_, j1_, j1, j2, j1_, j1]
    if j0%2 == 0:
        i_near[:] = [i0, i0, i0, i0, i1_, i1_]
    else:
        i_near[:] = [i0, i0, i0, i0, i1, i1]
    

    near = scan.ij2ind(i_near, j_near)
    near = near[(near >= 0) & (near < scan.N)]
    near = near[sel[near]]
    # n_near[k] = np.count_nonzero(near[(near >= 0) & (near < scan.N)])


    A = np.tensordot(C, M[near], axes=[[-1],[-2]]).transpose([2,0,1,3])
    B = np.tensordot(C, M[k], axes=[[-1],[-2]]).transpose([0,2,1])
    D = np.tensordot(A, B, axes=[[-1],[-2]]).transpose([1,3,0,2,4]).reshape(24*24,-1,3,3)
    b[k] = np.max(np.abs(np.trace(D, axis1=2, axis2=3)), axis=0)
    # b[k] = pyebsd.mis(M[near], np.dot(C, M[k]).transpose([0,2,1]), math='min', out='deg')
    # b[k] = np.max(np.trace(np.dot(np.dot(C, M[near]).transpose([2,0,1,3]), M[k].T).transpose([2,0,1,3]), axis1=2, axis2=3), axis=1)

     # np.dot(C, M[k]).transpose([0,2,1]), math='min', out='deg')
    a[k] = np.mean(b[k])

    if k%each == 0:
        sys.stdout.write('.')
        sys.stdout.flush()
    # a[k] = np.mean(pyebsd.mis(np.dot(R[k], C).transpose([1,2,0]), R[near], math='min', out='deg'))

print((time.time()-t0)/scan.N, time.time()-t0)
#     print i[k], j[k]
#     R[k]

# plt.imshow(n_near.reshape(32,-1), interpolation='nearest')
# plt.show()

a[np.isnan(a)] = 0.
ind_grid = scan.indices_grid
I = scan.gridify(np.arccos((a-1.)/2.)*180/np.pi)
I[np.isnan(I)] = -1.

d = [(c, len(c)) for c in b if c is not None]
d = np.array(d)
e = np.cumsum(d[:,1])
e = np.insert(e,0,0)
f = np.ndarray(e[-1])
for i in range(len(d)):
    f[e[i]:e[i+1]] = d[i,0]
f[f > 3.] = 3.
f = np.arccos((f-1.)/2.)*180/np.pi
_ = plt.hist(f, bins=200)

fig, ax = plt.subplots()
# ax.scatter(scan.x, scan.y, marker='H', c=a, lw=0, s=40)
ax.imshow(I, interpolation='nearest', extent=(np.min(scan.x), np.max(scan.x), np.min(scan.y), np.max(scan.y)))
ax.set_aspect('equal')
plt.show()

# for i in xrange(1,nrows-1):
#     for j in xrange(2, ncols-2, 2): 
#         R[i,j]
#         R[i,j-2], j-1, j+1, j+2
#         i-1: j-1, j+1
#         pass
#     for j in xrange(3, ncols-2, 2):
#         i : j-2, j-1, j+1, j+2
#         i+1: j-1, j+1
#         pass


# c = scan.gridify(scan.data, fill=np.repeat(-1,14))

# # load a selected area
# if 'sel' not in locals() or 'sel' not in globals() or reload_data is True:
#     # fname = os.path.join(pyebsd.DIR, 'data', 'QP170-375-30s_sel-grain.txt')
#     # sel = np.loadtxt(fname, dtype=bool)
#     sel = np.ndarray(len(data), dtype=bool)
#     sel[:] = True

# # plot IPF of the selected data
# ax2, img2 = scan.plot_IPF(tiling='hex', sel=(scan.ph==1) & (scan.CI>.2), gray=IQ)

# # average orientation of the austenite grain in the selected area
# R_avg = pyebsd.avg_orientation(R, sel=sel & (ph==2) & (CI>.2), vectorized=True)

# # parent_or = np.asarray([[1,-1,0],[1,1,-2],[1,1,1]]).T
# parent_or = np.diag([1,1,1]) # Identity matrix (canonical base)

# proj = [1,0,0]

# # plot PF of the selected data
# ax3 = pyebsd.plot_PF(R=R, sel=sel & (ph==1) & (CI>.2), proj=proj, contour=True, cmap=plt.get_cmap('Reds_r'))
# pyebsd.plot_PF(R=R, sel=sel & (ph==2) & (CI>.2), proj=proj, ax=ax3, contour=True, cmap=plt.get_cmap('Blues_r'))

# # plot PF aligned to orientation of the parent phase
# ax4 = pyebsd.plot_PF(R=R, sel=sel & (ph==1) & (CI>.2), proj=proj, contour=True, fill=True, cmap=plt.get_cmap('Reds_r'), parent_or=np.matmul(R_avg, parent_or))
# pyebsd.plot_PF(R=R, sel=sel & (ph==2) & (CI>.2), proj=proj, ax=ax4, contour=True, fill=True, cmap=plt.get_cmap('Blues_r'), parent_or=np.matmul(R_avg, parent_or))
# pyebsd.draw_std_traces(ax4, lw=.2)

# T_KS = pyebsd.OR() # K-S
# T_NW = pyebsd.OR(ds=[[1,1,0],[1,0,0]]) # N-W
# T_GT = pyebsd.OR(ds=[[5,12,17],[7,17,17]]) # G-T
# # TT_KS, TT_NW, TT_GT = T_KS.transpose([0,2,1]), T_NW.transpose([0,2,1]), T_GT.transpose([0,2,1])
# pyebsd.plot_PF(M=T_KS, mfc=[1,1,1,0], mec='k', mew=1, marker='o', ms=5, ax=ax4, proj=proj, verbose=False, parent_or=parent_or)
# pyebsd.plot_PF(M=T_NW, mfc=[1,1,1,0], mec='k', mew=1, marker='s', ms=5, ax=ax4, proj=proj, verbose=False, parent_or=parent_or)
# pyebsd.plot_PF(M=T_GT, mfc=[1,1,1,0], mec='k', mew=1, marker='v', ms=5, ax=ax4, proj=proj, verbose=False, parent_or=parent_or)
# pyebsd.draw_wulff_net(ax4, lw=.2)

# axis0, axis1 = np.asarray(parent_or)[:,0], np.asarray(parent_or)[:,1]
# ax4.annotate(uvw_label(axis0, s='\gamma'), xy=(1,0), ha='left', va='center')
# ax4.annotate(uvw_label(axis1, s='\gamma'), xy=(0,1), ha='center', va='bottom')

# plt.savefig('test_PF.pdf', bbox_inches='tight')

# plt.show()
