"""
Plotting overlaid EBSD maps
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

# Plot image quality (IQ) map using a gray color palette.
# colorbar=False is used to hide the colorbar
ipfmap = scan.plot_property(scan.IQ, cmap='gray', colorbar=False)

# Plot IPF map for only the fcc phase (ph == 2)
# colorfill='none' sets the background transparent
ipfmapfcc = scan.plot_IPF(sel=(scan.ph == 2), gray=scan.IQ, colorfill='none', ax=ipfmap.ax)

# Show plots
plt.show()
