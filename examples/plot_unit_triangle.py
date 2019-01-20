import numpy as np
import matplotlib.pyplot as plt
import pyebsd


fig, ax = plt.subplots()
pyebsd.unit_triangle(ax=ax)
ax.text(0, 1, ('Default color palette\n'
               'White spot in the barycenter of the unit triangle'),
        transform=ax.transAxes)

# --------------------------------------------------

whitespot = [24, 11, 42]

fig, ax = plt.subplots()
pyebsd.unit_triangle(ax=ax, whitespot=whitespot)
ax.text(0, 1, 'White spot in the direction {}'.format(whitespot),
        transform=ax.transAxes)

# --------------------------------------------------

whitespot = [48, 20, 83]

fig, ax = plt.subplots()
pyebsd.unit_triangle(ax=ax, whitespot=whitespot)
ax.text(0, 1, 'White spot in the direction {}'.format(whitespot),
        transform=ax.transAxes)

# --------------------------------------------------

whitespot = [2, 1, 3]

fig, ax = plt.subplots()
pyebsd.unit_triangle(ax=ax, whitespot=whitespot)
ax.text(0, 1, 'White spot in the direction {}'.format(whitespot),
        transform=ax.transAxes)

plt.show()
