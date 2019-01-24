"""
Plotting Kernel Average Misorientation (KAM)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pyebsd

fname = os.path.join('..', 'data', 'QP_bcc_fcc_single_austenite_grain.ang')
scan = pyebsd.load_ang_file(fname)
kammap = scan.plot_KAM(sel=(scan.ph == 1), vmax=2)
plt.show()
