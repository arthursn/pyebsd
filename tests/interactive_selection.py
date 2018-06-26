import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from builtins import input

import pyebsd

if 'fname_local' not in locals() or 'fname_local' not in globals():
    fname_local = ''

fname = os.path.join(pyebsd.DIR, 'data', 'QP170-375-30s.ang')
# fname = os.path.join(pyebsd.DIR, 'data', 'QP170-375-15_cropped.ang')
# fname = os.path.join(pyebsd.DIR, 'data', 'dummy.ang')
# fname = '/home/arthur/Documents/Tohoku2016/EBSD/ang/Q170-1m.ang'
# fname = '/home/arthur/Documents/Tohoku2016/EBSD/ang/QP170-375-0.ang'
# fname = '/home/arthur/Documents/Tohoku2016/EBSD/ang/QP170-375-15.ang'
# fname = '/home/arthur/Documents/Tohoku2016/EBSD/ang/QP170-450-30s.ang'
# fname = '/home/arthur/Documents/Tohoku2016/EBSD/ang/ADI375-15.ang'
# fname = '/home/arthur/qnaphg/Backup/EBSD/Tohoku2016/ang/ADI375-15.ang'
# fname = '/home/arthur/Documents/Tohoku2016/EBSD/ang/ADI300-15(2016-8-3).ang'

if fname != fname_local:
    fname_local = fname
    scan = pyebsd.Scandata(fname_local)

plt.ion()

ipf = scan.plot_IPF(sel=(scan.ph == 2) & (scan.CI > .2),
                    gray=scan.IQ, d=[1, 1, 0])
# ax, img = scan.plot_phase(gray=scan.IQ)
fig = plt.gcf()  # get current figure

# select points using lasso or rectangle selector. Interactive mode should be on
lasso = ipf.lasso_selector()
# lasso = ipf.rect_selector()

fig2, ax2 = plt.subplots()
pyebsd.set_tight_plt()
fig3, ax3 = plt.subplots(facecolor='white', figsize=(5, 5))
pyebsd.set_tight_plt()

proj = [1, 0, 0]

key = ''
while key is not 'q':
    if np.count_nonzero(ipf.sel) > 0:
        R_fcc = pyebsd.avg_orientation(scan.R, sel=ipf.sel & (scan.ph == 2) & (
            scan.CI > .2), vect=False, verbose=True)  # , plot=True, n=10, maxdev=10., it=5)

        # plot IPF of the selected data
        ax2.cla()  # clear axis
        scan.plot_IPF(sel=ipf.sel & (scan.ph == 1) & (
            scan.CI > .3), gray=scan.IQ, tiling='hex', ax=ax2)

        # plot PF of the selected data
        ax3.cla()  # clear axis
        ax3.set_aspect('equal')
        ax3.axis('off')
        pyebsd.draw_circle_frame(ax3, lw=.5)
        pyebsd.draw_std_traces(ax3, lw=.5)
        scan.plot_PF(sel=ipf.sel & (scan.ph == 1) & (scan.CI > .2), proj=proj,
                     contour=True, cmap=plt.get_cmap('Reds_r'), ax=ax3, parent_or=R_fcc)
        scan.plot_PF(sel=ipf.sel & (scan.ph == 2) & (scan.CI > .2), proj=proj,
                     contour=True, cmap=plt.get_cmap('Blues_r'), ax=ax3, parent_or=R_fcc)

        plt.draw_all()

    key = input(
        'Press ENTER to process selected data or type q to quit selection mode ')

lasso.disconnect()
plt.close('all')

if np.count_nonzero(ipf.sel) > 0:
    sel = ipf.sel
    exec(open('accurate_OR.py').read())
