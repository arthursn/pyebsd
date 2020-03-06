"""
Plotting EBSD maps
"""

import matplotlib.pyplot as plt
import pyebsd


import os
# .ang file available in the data folder
# Replace angfile variable with the path to your .ang file
angfile = os.path.join('..', 'data', 'ADI_bcc_fcc.ang')

# load ebsd scandata
# So far, pyebsd only supports loading .ang files generated
# by the TSL OIM software
scan = pyebsd.load_scandata(angfile)

# Plot inverse pole figure map
# gray is used to set the quality index as grayscale
ipfmap = scan.plot_IPF(gray=scan.IQ)

# Plot IPF map for only the fcc phase (ph == 2)
ipfmapfcc = scan.plot_IPF(sel=(scan.ph == 2), gray=scan.IQ, tiling='hex')

# Plot phase map
phmap = scan.plot_phase(gray=scan.IQ)

ipfmap.ax.plot(scan.x, scan.y, 'r.', ms='1')
ipfmapfcc.ax.plot(scan.x, scan.y, 'r.', ms='1')
phmap.ax.plot(scan.x, scan.y, 'k.', ms='1')
plt.ion()
plt.show()

phmap.lasso_selector()

key = input('Type ENTER')

p1 = scan.plot_IPF(sel=(scan.ph == 2) & phmap.sel, gray=scan.IQ, tiling='hex')
p2 = scan.plot_IPF(sel=(scan.ph == 2) & phmap.sel, gray=scan.IQ, tiling='rect')
p3 = scan.plot_phase(sel=phmap.sel, gray=scan.IQ, tiling='hex')
p4 = scan.plot_phase(sel=phmap.sel, gray=scan.IQ, tiling='rect')
p1.ax.plot(scan.x[(scan.ph == 2) & phmap.sel], scan.y[(scan.ph == 2) & phmap.sel], 'r.', ms='1')
p2.ax.plot(scan.x[(scan.ph == 2) & phmap.sel], scan.y[(scan.ph == 2) & phmap.sel], 'r.', ms='1')
p3.ax.plot(scan.x[phmap.sel], scan.y[phmap.sel], 'k.', ms='1')
p4.ax.plot(scan.x[phmap.sel], scan.y[phmap.sel], 'k.', ms='1')

# Show plots