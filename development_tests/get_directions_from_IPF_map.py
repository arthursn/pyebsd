import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyebsd
from PIL import Image


def get_gray_from_color(rgb):
    if isinstance(rgb, (list, tuple)):
        rgb = np.array(rgb)

    shape = rgb.shape
    ndim = rgb.ndim
    if ndim != 2:
        rgb = rgb.reshape(-1, 3)

    gray = rgb.max(axis=1)/255

    if ndim != 2:
        gray = gray.reshape(shape[:-1])

    return gray


def get_IPF_from_color(rgb, **kwargs):
    """
    Get the IPF direction(s) from the corresponding list color
    """
    if isinstance(rgb, (list, tuple)):
        rgb = np.array(rgb)

    shape = rgb.shape
    ndim = rgb.ndim
    if ndim != 2:
        rgb = rgb.reshape(-1, 3)

    # whitespot: white spot in the unit triangle
    # By default, whitespot is in the barycenter of the unit triangle
    whitespot = kwargs.pop('whitespot', [0.48846011, 0.22903335, 0.84199195])
    pwr = kwargs.pop('pwr', .75)

    # Select variant where w >= u >= v
    whitespot = np.sort(whitespot)
    whitespot = whitespot[[1, 0, 2]]

    kR = whitespot[2] - whitespot[0]
    kG = whitespot[0] - whitespot[1]
    kB = whitespot[1]

    v = kB*rgb[:, 2]**(1./pwr)
    u = kG*rgb[:, 1]**(1./pwr) + v
    w = kR*rgb[:, 0]**(1./pwr) + u

    uvw = np.array([u, v, w])
    uvwnorm = np.linalg.norm(uvw, axis=0)
    uvw = (uvw/uvwnorm).T

    if ndim != 2:
        uvw = uvw.reshape(shape)

    return uvw


def reconstruct_IPF_map(fname):
    img = Image.open(fname)
    w, h = img.size

    rgb = np.array(img)[:, :, :3]  # drop alpha channel

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

    ax1.imshow(rgb)
    ax1.axis('off')
    ax1.set_title('Original figure')

    uvw = get_IPF_from_color(rgb)
    gray = get_gray_from_color(rgb)

    rgb1 = pyebsd.get_color_IPF(uvw)
    rgb1 = (rgb1.T*gray.T).T.astype(np.uint8)

    ax2.imshow(rgb1)
    ax2.axis('off')
    ax2.set_title('Reconstructed figure by first calculating\n'
                  'directions and then converting to colors')

    diffmap = ax3.imshow((((rgb.astype(int) - rgb1.astype(int))**2.).sum(axis=2))
                         ** .5, vmin=0, vmax=2, cmap='magma')
    ax3.axis('off')
    ax3.set_title('Difference (squared residuals)')
    cbar = fig.colorbar(diffmap, ax=ax3, extend='both')

    fig.tight_layout()

# def reconstruct_orientations(fname1, d1, fname2, d2):
#     fname1, d1 = 'IPF_100_gray_IQ.png', [1, 0, 0]
#     fname2, d2 = 'IPF_001_gray_IQ.png', [0, 0, 1]

#     img1 = Image.open(fname1)
#     img2 = Image.open(fname2)

#     if img1.size != img2.size:
#         raise Exception('Images have different sizes')

#     w, h = img1.size

#     rgb1 = np.array(img1)[:, :, :3].reshape(-1, 3)
#     rgb2 = np.array(img2)[:, :, :3].reshape(-1, 3)

#     ipf1 = get_IPF_from_color(rgb1)
#     ipf2 = get_IPF_from_color(rgb2)

#     C = pyebsd.list_cubic_symmetry_operators()

#     A = np.array([d1, d2, np.cross(d1, d2)])
#     B = np.array([ipf1, ipf2, np.cross(ipf1, ipf2)]).transpose(1, 0, 2)
#     R = np.dot(B, A).transpose(0, 1, 2)

#     x, y = np.mgrid[:w, :h]

#     data = pd.DataFrame(columns=['x', 'y', 'phi1', 'Phi', 'phi2', 'ph', 'IQ'])
#     data['phi1'], data['Phi'], data['phi2'] = pyebsd.rotation_matrix_to_euler_angles(R)
#     data['x'], data['y'] = x.ravel(), y.ravel()
#     data['IQ'] = get_gray_from_color(rgb1)
#     data.fillna(1, inplace=True)
#     scan = pyebsd.ScanData(data, 'SqrGrid', 1, 1, w, w, h)


if __name__ == '__main__':
    reconstruct_IPF_map('../data/unit_triangle.png')
    reconstruct_IPF_map('IPF_100_gray_IQ.png')

    plt.show()
