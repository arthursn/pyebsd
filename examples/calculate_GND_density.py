# -*- coding: utf-8 -*-

"""
[1] C. Moussa, M. Bernacki, R. Besnard, N. Bozzolo, Ultramicroscopy 179 (2017) 63–72.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyebsd
import random

if __name__ == '__main__':
    scan = pyebsd.load_scandata(os.path.join('..', 'data', 'ADI_bcc_fcc_cropped.ang'))

    distance = []
    kam = []
    convention, maxmis, nneighbors = 'OIM', 5., 5
    # convention, maxmis, nneighbors = 'fixed', 5., 15

    for d in range(1, nneighbors + 1):
        distance.append(scan.get_distance_neighbors(distance=d, distance_convention=convention))
        kam.append(scan.get_KAM(distance=d, distance_convention=convention,
                                sel=(scan.ph == 1), maxmis=maxmis))

    distance = np.array(distance)
    kam = np.vstack(kam).T
    kamavg = kam.mean(axis=0)

    # Linear fit of KAM vs distance
    N = len(distance)
    # Slope m
    m = (N*(kam*distance).sum(axis=1) - kam.sum(axis=1)*distance.sum())
    m /= (N*(distance**2).sum() - (distance.sum())**2)
    # Intercept b
    b = kam.sum(axis=1) - m*distance.sum()
    b /= N

    kamfit = m.reshape(-1, 1)*distance + b.reshape(-1, 1)
    # Standard deviation
    sd = ((kam - kamfit)**2).sum(axis=1)
    sd /= N
    sd **= .5

    kammean = kam.mean(axis=1).reshape(-1, 1)
    SStot = ((kam - kammean)**2.).sum(axis=1)
    SStot[SStot == 0.] = 1.
    SSres = ((kam - kamfit)**2.).sum(axis=1)
    # R squared
    Rsquared = 1. - SSres/SStot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    selection = (scan.ph == 1)
    # Plot map of the slope
    mmap = scan.plot_property(m, sel=selection, ax=ax1, tiling='hex', vmin=0)
    mmap.cax.set_label(u'KAM gradient (°/μm)')
    ax1.set_title('KAM gradient (proportional to GND density) vs distance fit')

    indices, = np.where(selection)

    # Plot KAM vs distance for a random pixel
    tracker, = mmap.ax.plot([], [], 'rx')
    raw, = ax2.plot(distance, kamavg, 'kx')
    fit, = ax2.plot(distance, np.polyval(np.polyfit(distance, kamavg, 1), distance), 'r-')
    txt = ax2.text(.98, .98, 'Average KAM', ha='right', va='top', transform=ax2.transAxes)
    ax2.set_xlabel(u'Distance (μm)')
    ax2.set_ylabel(u'KAM (°)')
    ax2.set_title('Click on the map on the left to plot KAM vs distance')
    ax2.set_xlim(-.2, max(distance) + .2)
    ax2.set_ylim(-.2, maxmis + .2)

    def plot_kam_vs_distance(event):
        if event.inaxes == ax1:
            try:
                # i, j coordinates of the click event
                i = int(round(event.ydata/scan.dy))
                j = int(2*round(event.xdata/scan.dx))
                idx = scan.ij_to_index(i, j)
            except:
                pass

            if idx in indices:
                txt.set_text(u'slope={:g}, intercept={:g}, R²={:g}'.format(m[idx], b[idx], Rsquared[idx]))
                tracker.set_data([scan.x[idx]], [scan.y[idx]])
                raw.set_data(distance, kam[idx])
                fit.set_data(distance, kamfit[idx])

                fig.canvas.draw()
                fig.canvas.flush_events()

    cid = fig.canvas.mpl_connect('button_press_event', plot_kam_vs_distance)

    # Reset figure borders
    fig.tight_layout()

    plt.show()
