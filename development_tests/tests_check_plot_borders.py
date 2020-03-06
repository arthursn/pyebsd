"""
Plotting EBSD maps
"""

from builtins import input
import matplotlib.pyplot as plt
import pyebsd


import os
# .ang file available in the data folder
# Replace angfile variable with the path to your .ang file
angfile = os.path.join('..', 'data', 'ADI_bcc_fcc.ang')

# load ebsd scandata
scan = pyebsd.load_scandata(angfile)

# Plot inverse pole figure map
ipfmap = scan.plot_IPF(gray=scan.IQ)
ipfmap.ax.plot(scan.x, scan.y, 'r.', ms='1')

# Plot IPF map for only the fcc phase (ph == 2)
ipfmapfcc = scan.plot_IPF(sel=(scan.ph == 2), gray=scan.IQ, tiling='hex')
ipfmapfcc.ax.plot(scan.x[scan.ph == 2], scan.y[scan.ph == 2], 'r.', ms='1')

# Plot phase map
phmap = scan.plot_phase(gray=scan.IQ)
phmap.ax.plot(scan.x, scan.y, 'k.', ms='1')

# Show plots
plt.ion()
plt.show()

phmap.lasso_selector()
# phmap.rect_selector()

key = input('Type ENTER')

p1 = scan.plot_IPF(sel=(scan.ph == 2) & phmap.sel, gray=scan.IQ, tiling='hex')
p1.ax.plot(scan.x[(scan.ph == 2) & phmap.sel], scan.y[(scan.ph == 2) & phmap.sel], 'r.', ms='1')

p2 = scan.plot_IPF(sel=(scan.ph == 2) & phmap.sel, gray=scan.IQ, tiling='rect')
p2.ax.plot(scan.x[(scan.ph == 2) & phmap.sel], scan.y[(scan.ph == 2) & phmap.sel], 'r.', ms='1')

p3 = scan.plot_phase(sel=phmap.sel, gray=scan.IQ, tiling='hex')
p3.ax.plot(scan.x[phmap.sel], scan.y[phmap.sel], 'k.', ms='1')

p4 = scan.plot_phase(sel=phmap.sel, gray=scan.IQ, tiling='rect')
p4.ax.plot(scan.x[phmap.sel], scan.y[phmap.sel], 'k.', ms='1')
