"""
Export selection as ang file
"""

import matplotlib.pyplot as plt
import pyebsd

from builtins import input
import os

angfile = os.path.join("..", "data", "ADI_bcc_fcc.ang")
scan = pyebsd.load_scandata(angfile)

# enable matplotlib interactive mode
plt.ion()
# plot IPF
ipf = scan.plot_IPF(gray=scan.IQ, sel=scan.ph == 2)
# enable lasso selector
ipf.lasso_selector()

key = input("Press ENTER to process selected data ")
# disable interactive mode
plt.ioff()

# plot ipf of the selection
ipfsel = scan.plot_IPF(sel=ipf.sel, gray=scan.IQ)

# save selection as selection.ang
scan.save_ang_file("selection.ang", ipf.sel)
# This also works:
# pyebsd.save_ang_file('selection.ang', scan, ipf.sel)

# load selection.ang
scansel = pyebsd.load_scandata("selection.ang")
# plot ipf from selection
ipfscansel = scansel.plot_IPF(gray=scansel.IQ)

plt.show()
