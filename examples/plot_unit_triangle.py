import numpy as np
import matplotlib.pyplot as plt
import pyebsd


fig, ax = plt.subplots()

whitespot = [48, 20, 83]
pyebsd.unit_triangle(ax=ax)

# xp, yp = pyebsd.stereographic_projection(whitespot)
# ax.plot(xp, yp, 'kx')

ax.text(0, 1, ('Default color palette\n'
               'White spot in the direction [48, 20, 83]'),
        transform=ax.transAxes)


fig, ax = plt.subplots()

whitespot = [2, 1, 3]
pyebsd.unit_triangle(ax=ax, whitespot=whitespot)

# xp, yp = pyebsd.stereographic_projection(whitespot)
# ax.plot(xp, yp, 'kx')

ax.text(0, 1, 'White spot in the direction [2, 1, 3]',
        transform=ax.transAxes)

plt.show()
