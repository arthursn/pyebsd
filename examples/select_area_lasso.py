"""
Lasso selector to select areas
"""

import matplotlib.pyplot as plt
import pyebsd

from builtins import input
import os
# .ang file available in the data folder
# Replace angfile variable with the path to your .ang file
angfile = os.path.join('..', 'data', 'ADI_bcc_fcc.ang')

# load ebsd scandata
# So far, pyebsd only supports loading .ang files generated
# by the TSL OIM software
scan = pyebsd.load_scandata(angfile)

# Enable matplotlib.pyplot interactive node, which means
# that the plots open automatically and do not interrupt
# the python interpreter.
# If plt.ion() is not called, you must to run plt.show()
# after calling pyebsd plot functions.
plt.ion()


# Plot IPF map for only the fcc phase (ph == 2)
ipf = scan.plot_IPF(sel=(scan.ph == 2), gray=scan.IQ)

# Enable lasso_selector. Click with the mouse in the open IPF
# plot and select an area.
# Use the left button to create a new vertex and the right button
# to close the loop.
# When finished, press ENTER in the
# terminal/command prompt
ipf.lasso_selector()

key = input('Press ENTER to process selected data ')

# Plot IPF map of the selected area
ipfselection = scan.plot_IPF(sel=ipf.sel, gray=scan.IQ)

# Plot pole figure of the bcc phase in the selected area
pfbcc = scan.plot_PF(sel=ipf.sel & (scan.ph == 1), contour=True)
pfbcc.set_title('Pole Figure bcc phase')

# Plot pole figure of the fcc phase in the selected area
pffcc = scan.plot_PF(sel=ipf.sel & (scan.ph == 2), contour=True)
pffcc.set_title('Pole Figure fcc phase')
