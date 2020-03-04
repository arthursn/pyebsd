import os
import numpy as np
import matplotlib.pyplot as plt
import pyebsd

fname = os.path.join('..', 'data', 'QP_bcc_fcc_single_austenite_grain.ang')
scan = pyebsd.load_ang_file(fname)

distance = []
kam = []
for d in range(1, 6):
	distance.append(scan.get_distance_neighbors(d, distance_convention='OIM'))
	kam.append(scan.get_KAM(distance=d, maxmis=5, distance_convention='OIM', sel=(scan.ph == 1)).mean())

plt.plot(distance, kam, 'kx')
plt.xlim(0)
plt.ylim(0)
plt.show()
