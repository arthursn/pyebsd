import matplotlib.pyplot as plt
import pyebsd

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
ax1, ax2, ax3, ax4 = axes.ravel()

pyebsd.unit_triangle(ax=ax1)
ax1.text(
    0,
    1,
    ("Default color palette\n" "White spot in the barycenter of the unit triangle"),
    transform=ax1.transAxes,
)

# --------------------------------------------------

whitespot = [24, 11, 42]

pyebsd.unit_triangle(ax=ax2, whitespot=whitespot)
ax2.text(
    0, 1, "White spot in the direction {}".format(whitespot), transform=ax2.transAxes
)

# --------------------------------------------------

whitespot = [48, 20, 83]

pyebsd.unit_triangle(ax=ax3, whitespot=whitespot)
ax3.text(
    0, 1, "White spot in the direction {}".format(whitespot), transform=ax3.transAxes
)

# --------------------------------------------------

whitespot = [2, 1, 3]

pyebsd.unit_triangle(ax=ax4, whitespot=whitespot)
ax4.text(
    0, 1, "White spot in the direction {}".format(whitespot), transform=ax4.transAxes
)

fig.set_tight_layout("tight")
plt.show()
