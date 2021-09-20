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
ipfmapfcc = scan.plot_IPF(sel=(scan.ph == 2), gray=scan.IQ)

# Plot phase map
phmap = scan.plot_phase(gray=scan.IQ)

# Show plots
plt.show()
