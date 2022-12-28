# -*- coding: utf-8 -*-

"""
[1] C. Moussa, M. Bernacki, R. Besnard, N. Bozzolo, Ultramicroscopy 179 (2017) 63–72.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pyebsd

if __name__ == "__main__":
    scan = pyebsd.load_scandata(os.path.join("..", "data", "ADI_bcc_fcc_cropped.ang"))
    selection = scan.ph == 1  # Selects phase indexed as 1
    (indices,) = np.where(selection)

    distance = []  # real distance in (normally um) to n-th nearest neighbor
    kam = []  # list with KAM values for every pixel for each distance
    convention, maxmis, nneighbors = "OIM", 5.0, 5
    # convention, maxmis, nneighbors = 'fixed', 5., 15

    for d in range(1, nneighbors + 1):
        distance.append(
            scan.get_distance_neighbors(distance=d, distance_convention=convention)
        )
        kam.append(
            scan.get_KAM(
                distance=d, distance_convention=convention, sel=selection, maxmis=maxmis
            )
        )

    # Converts lists to numpy array
    distance = np.array(distance)
    kam = np.array(kam).T  # more convenient to work with the transposed array

    # Average KAM values for each DISTANCE
    kamavg = np.nan_to_num(kam).mean(axis=0)  # nan values become 0 first

    # Linear fit of KAM vs distance
    # Slope m
    m = nneighbors * (kam * distance).sum(axis=1) - kam.sum(axis=1) * distance.sum()
    m /= nneighbors * (distance**2).sum() - (distance.sum()) ** 2
    # Intercept b
    b = kam.sum(axis=1) - m * distance.sum()
    b /= nneighbors

    # Fitted KAM values
    kamfit = m.reshape(-1, 1) * distance + b.reshape(-1, 1)
    # Standard deviation
    sd = ((kam - kamfit) ** 2).sum(axis=1)
    sd /= nneighbors
    sd **= 0.5

    kammean = kam.mean(axis=1).reshape(-1, 1)
    SStot = ((kam - kammean) ** 2.0).sum(axis=1)
    SStot[SStot == 0.0] = 1.0
    SSres = ((kam - kamfit) ** 2.0).sum(axis=1)
    # R squared
    Rsquared = 1.0 - SSres / SStot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot map of the slope
    mmap = scan.plot_property(m, sel=selection, ax=ax1, tiling="hex", vmin=0)
    mmap.cax.set_label("KAM gradient (°/μm)")
    ax1.set_title("Slope of KAM (proportional to GND density) vs distance")

    # Plot kamavg vs distance
    (tracker,) = mmap.ax.plot([], [], "rx")
    (raw,) = ax2.plot(distance, kamavg, "kx")
    (fit,) = ax2.plot(
        distance, np.polyval(np.polyfit(distance, kamavg, 1), distance), "r-"
    )
    txt = ax2.text(
        0.98, 0.98, "Average KAM", ha="right", va="top", transform=ax2.transAxes
    )
    ax2.set_xlabel("Distance (μm)")
    ax2.set_ylabel("KAM (°)")
    ax2.set_title("Click on the map on the left to plot KAM vs distance")
    ax2.set_xlim(-0.2, max(distance) + 0.2)
    ax2.set_ylim(-0.2, maxmis + 0.2)

    # Callback to button press event
    def plot_kam_vs_distance(event):
        if event.inaxes == ax1:
            try:
                idx = scan.xy_to_index(event.xdata, event.ydata)
            except:
                pass

            if idx in indices:
                txt.set_text(
                    "slope={:g}, intercept={:g}, R²={:g}".format(
                        m[idx], b[idx], Rsquared[idx]
                    )
                )
                tracker.set_data([scan.x[idx]], [scan.y[idx]])
                raw.set_data(distance, kam[idx])
                fit.set_data(distance, kamfit[idx])

                fig.canvas.draw()
                fig.canvas.flush_events()

    cid = fig.canvas.mpl_connect("button_press_event", plot_kam_vs_distance)

    # Reset figure borders
    fig.tight_layout()

    plt.show()
