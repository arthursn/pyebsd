import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from builtins import input

import pyebsd

if 'fname_local' not in locals() or 'fname_local' not in globals():
    fname_local = ''

# fname = os.path.join('..', 'data', 'ADI_bcc_fcc.ang')
fname = os.path.join('..', 'data', 'QP_bcc_fcc_single_austenite_grain.ang')

basename = os.path.basename(fname)
basename = os.path.splitext(basename)[0]

if fname != fname_local:
    fname_local = fname
    scan = pyebsd.load_scandata(fname_local)

plt.ion()

ipf0 = scan.plot_IPF(gray=scan.IQ, d=[0, 0, 1])
# ipf0.fig.savefig(basename + '_IPF.pdf', bbox_inches='tight', pad_inches=0)
plt.close(ipf0.fig)

phmap = scan.plot_phase(gray=scan.IQ)
# phmap.fig.savefig(basename + '_ph.pdf', bbox_inches='tight', pad_inches=0)
plt.close(phmap.fig)

ipf = scan.plot_IPF(sel=(scan.ph == 2) & (scan.CI > .2),
                    gray=scan.IQ, d=[0, 0, 1])
fig = ipf.fig
# fig.savefig(basename + '_IPF_fcc.pdf', bbox_inches='tight', pad_inches=0)

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
        M_fcc = pyebsd.avg_orientation(scan.M, sel=ipf.sel & (scan.ph == 2) & (
            scan.CI > .2), vectorized=False, verbose=True)  # , plot=True, n=10, maxdev=10., it=5)

        # plot IPF of the selected data
        ax2.cla()  # clear axis
        scan.plot_IPF(sel=ipf.sel, gray=scan.IQ, tiling='hex', ax=ax2)

        # plot PF of the selected data
        ax3.cla()  # clear axis
        ax3.set_aspect('equal')
        ax3.axis('off')
        pyebsd.draw_circle_frame(ax3, lw=.5)
        pyebsd.draw_std_traces(ax3, lw=.5)
        scan.plot_PF(sel=ipf.sel & (scan.ph == 1) & (scan.CI > .2), proj=proj,
                     contour=True, cmap=plt.get_cmap('Reds_r'), ax=ax3, rotation=M_fcc.T)
        scan.plot_PF(sel=ipf.sel & (scan.ph == 2) & (scan.CI > .2), proj=proj,
                     contour=True, cmap=plt.get_cmap('Blues_r'), ax=ax3, rotation=M_fcc.T)

        plt.draw_all()

    key = input('Press ENTER to process selected data or type q to quit selection mode ')

# fig.savefig(basename + '_IPF_fcc_sel.pdf', bbox_inches='tight', pad_inches=0)
# fig2.savefig(basename + '_IPF_sel.pdf')
# fig3.savefig(basename + '_PF_bcc_fcc.pdf', bbox_inches='tight', pad_inches=0)

lasso.disconnect()
plt.close('all')

if np.count_nonzero(ipf.sel) > 0:
    sel = ipf.sel
    exec(open('accurate_OR.py').read())
