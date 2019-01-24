import numpy as np
import matplotlib.pyplot as plt

import pyebsd
from pyebsd import get_color_IPF, toimage, stereographic_projection


def plot_unit_triangle(ax=None, n=512, **kwargs):
    """
    Only valid for cubic system
    """
    # x and y max values in the stereographic projection corresponding to
    # the unit triangle
    xmax, ymax = 2.**.5 - 1, (3.**.5 - 1)/2.

    # map n x n square around unit triangle
    xp, yp = np.meshgrid(np.linspace(0, xmax, n), np.linspace(0, ymax, n))
    xp, yp = xp.ravel(), yp.ravel()
    # convert projected coordinates (xp, yp) to uvw directions
    u, v, w = 2*xp, 2*yp, 1-xp**2-yp**2
    uvw = np.vstack([u, v, w]).T

    col = np.ndarray(uvw.shape)
    # select directions that will fit inside the unit triangle, i.e.,
    # only those where w >= u >= v
    sel = (w >= u) & (u >= v)
    # uvw directions to corresponding color
    col[sel] = get_color_IPF(uvw[sel], **kwargs)
    # fill points outside the unit triangle in white
    col[~sel] = [255, 255, 255]

    # Centroid calculated by numerical method
    print(pyebsd.stereographic_projection_to_direction([xp[sel].mean(), yp[sel].mean()]))

    img = toimage(col.reshape(n, n, 3))

    if ax is None:
        fig, ax = plt.subplots(facecolor='white')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.imshow(img, interpolation='None', origin='lower',
              extent=(0, xmax, 0, ymax))

    # Draw borders of unit triangle
    t = np.linspace(0, 1., n)

    # [np.repeat(1.,n), t, np.repeat(1.,n)]
    # [t[::-1], t[::-1], np.repeat(1.,n)]
    # [t, np.repeat(0,n), np.repeat(1.,n)]
    u = np.hstack([np.repeat(1., n), t[::-1], t])
    v = np.hstack([t, t[::-1], np.repeat(0, n)])
    w = np.hstack([np.repeat(1., n), np.repeat(1., n), np.repeat(1., n)])
    x, y = stereographic_projection([u, v, w])
    ax.plot(x, y, 'k-', lw=2)

    ax.annotate('001', xy=stereographic_projection([0, 0, 1]), xytext=(
        0, -10), textcoords='offset points', ha='center', va='top', size=30)
    ax.annotate('101', xy=stereographic_projection([1, 0, 1]), xytext=(
        0, -10), textcoords='offset points', ha='center', va='top', size=30)
    ax.annotate('111', xy=stereographic_projection([1, 1, 1]), xytext=(
        0, 10), textcoords='offset points', ha='center', va='bottom', size=30)
    ax.set_xlim(-.01, xmax+.01)
    ax.set_ylim(-.01, ymax+.01)

    return ax


# Barycenter half circular cap
# integrate (2-(x+1)^2)^0.5 from (3^0.5-1)/2 to 2^0.5 - 1
Ac = (np.pi - 3)/12  # area
# integrate x*(2-(x+1)^2)^0.5 from (3^0.5-1)/2 to 2^0.5 - 1
AcCxc = (-2 + 3*3**.5 - np.pi)/12  # area times Cx
# integrate y*((2-y^2)^0.5 - 1 - (3^0.5-1)/2 ) from 0 to (3^0.5-1)/2
# or
# integrate (2-(x+1)^2)/2 from (3^0.5-1)/2 to 2^0.5 - 1
AcCyc = (-7 + 16*2**.5 - 9*3**.5)/24  # area times Cy

# Barycenter isosceles right triangle
At = (2 - 3**.5)/4  # area
Cxt = (3**.5 - 1)/3  # Cx
Cyt = (3**.5 - 1)/6  # Cy

# Calculate barycenter by decomposition
A = Ac + At
Cx = (AcCxc + At*Cxt)/A
Cy = (AcCyc + At*Cyt)/A

whitespot = pyebsd.stereographic_projection_to_direction([Cx, Cy])

print(whitespot)

fig, ax = plt.subplots()

plot_unit_triangle(ax=ax, n=512, whitespot=whitespot)

ax.plot(AcCxc/Ac, AcCyc/Ac, 'kx')
ax.plot(Cxt, Cyt, 'kx')
ax.plot(Cx, Cy, 'kx')

ax.axvline((3**.5 - 1)/2, ls='--', color='k')

plt.show()
