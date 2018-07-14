import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from builtins import input

import pyebsd

try:
    scan
    R_bcc
    M_bcc

    R_fcc
    M_fcc
    M_fcc_avg
    R_fcc_avg

    C

    plt.ion()
except:
    fname = os.path.join(pyebsd.DIR, 'data', 'QP170-375-15_cropped.ang')
    scan = pyebsd.Scandata(fname)

    sel_fcc = (scan.ph == 2) & (scan.CI > .2)
    R_fcc = scan.R[sel_fcc]
    M_fcc = R_fcc.transpose([0, 2, 1])
    R_fcc_avg = pyebsd.avg_orientation(R_fcc,
                                       vectorized=False, verbose=True,
                                       n=10, maxdev=10., it=5)
    M_fcc_avg = R_fcc_avg.T

    sel_bcc = (scan.ph == 1) & (scan.CI > .2)
    R_bcc = scan.R[sel_bcc]
    M_bcc = R_bcc.transpose([0, 2, 1])

    C = pyebsd.list_symm()
    # # 4 dimensional numpy narray(N,24,3,3)
    # Mprime = np.tensordot(C, M_bcc,
    #                       axes=[[-1], [-2]]).transpose([2, 0, 1, 3])

# fig1, ax1 = plt.subplots()
# pyebsd.initialize_frame_PF(ax1)
# pyebsd.plot_PF(R_bcc, ax=ax1, contour=True, cmap='Reds_r', proj=[1, 1, 0])
# pyebsd.plot_PF(R_fcc, ax=ax1, contour=True, cmap='Blues_r', proj=[1, 1, 1])

# fig2, ax2 = plt.subplots()
# pyebsd.initialize_frame_PF(ax2)
# pyebsd.plot_PF(R_bcc, ax=ax2, contour=True, cmap='Reds_r', proj=[1, 1, 1])
# pyebsd.plot_PF(R_fcc, ax=ax2, contour=True, cmap='Blues_r', proj=[1, 1, 0])


fig, ax = plt.subplots()
pyebsd.initialize_frame_PF(ax)

T_KS = pyebsd.OR()
# T_KS^-1 = T_KS^T
T_KS_T = T_KS.transpose([0, 2, 1])

M_KS = np.dot(T_KS, M_fcc_avg)
R_KS = M_KS.transpose(0, 2, 1)


np.tensordot(C, M_bcc,
             axes=[[-1], [-2]]).transpose([2, 0, 1, 3])

# T_KS^T . M_bcc
M_KS_all = np.tensordot(C, M_KS,
                        axes=[[-1], [-2]]).transpose([2, 0, 1, 3])

R_fcc_inv = np.tensordot(T_KS_T, M_KS_all.reshape(-1, 3, 3),
                         axes=[[-1], [-2]]).transpose([2, 0, 3, 1])

# M_bcc_all = np.tensordot(C, M_bcc,
#                          axes=[[-1], [-2]]).transpose([2, 0, 1, 3])
# # Flatten M_bcc_all
# M_bcc_all = M_bcc_all.reshape(-1, 3, 3)
R_fcc_inv_exp = np.tensordot(T_KS_T, M_bcc,
                             axes=[[-1], [-2]]).transpose([2, 0, 3, 1])

# Flatten R_fcc_inv*
R_fcc_inv = R_fcc_inv.reshape(-1, 3, 3)
R_fcc_inv_exp = R_fcc_inv_exp.reshape(-1, 3, 3)

pyebsd.plot_PF(R_fcc_inv_exp, ax=ax, contour=True)
pyebsd.plot_PF(R_fcc_inv, ax=ax, marker='.', color='r', ms=2)
pyebsd.plot_PF(R_fcc_avg, ax=ax, marker='x', ms=5, color='r')
