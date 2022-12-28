"""
Plotting overlaid EBSD maps
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pyebsd

# .ang file available in the data folder
# Replace angfile variable with the path to your .ang file
angfile = Path(__file__).parent.parent / "data" / "ADI_bcc_fcc.ang"

# load ebsd scandata
# So far, pyebsd only supports loading .ang files generated
# by the TSL OIM software
scan = pyebsd.load_scandata(angfile)

# Plot image quality (IQ) map using a gray color palette.
# colorbar=False is used to hide the colorbar
iqmap = scan.plot_property(scan.IQ, cmap="gray", colorbar=False)

# Plot IPF map for only the fcc phase (ph == 2)
# colorfill='none' sets the background transparent
ipfmap = scan.plot_IPF(sel=(scan.ph == 2), gray=scan.IQ, colorfill="none", ax=iqmap.ax)

# Show plots
plt.show()
