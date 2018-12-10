import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from itertools import permutations

from .orientation import euler_rotation, PF, IPF
from ..crystal import stereographic_projection
from ..draw import modify_show, set_tight_plt, \
    draw_circle_frame, toimage, ScaleBar
from ..selection import LassoSelector2, RectangleSelector2


class EBSDMap(object):
    """
    Documentation
    """

    def __init__(self, x, y, img, ax, fig):
        self.x = x
        self.y = y
        self.img = img
        self.ax = ax
        self.fig = fig

        self._selector = None

    @property
    def sel(self):
        if self.selector is not None:
            return self.selector.sel

    @property
    def selector(self):
        return self._selector

    @selector.setter
    def selector(self, selector_widget):
        if self._selector is not None:
            try:
                self._selector.clear()
            except:
                pass
            self._selector.disconnect()
        self._selector = selector_widget

    def lasso_selector(self, lineprops=dict(color='white')):
        self.selector = LassoSelector2(
            self.ax, self.x, self.y, lineprops=lineprops)
        return self.selector

    def rect_selector(self, rectprops=dict(edgecolor='white', fill=False),
                      aspect=None):
        self.selector = RectangleSelector2(
            self.ax, self.x, self.y, rectprops=rectprops, aspect=aspect)
        return self.selector


def _calculate_barycenter_unit_triangle():
    from ..crystal import projection_to_direction

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

    return projection_to_direction([Cx, Cy])


def get_color_IPF(uvw, **kwargs):
    """
    Get the IPF color(s) of a given uvw direction or list of directions.
    So far it is only valid for cubic system. 
    Because of the symmetry in the cubic system, [u,v,w], [v,u,w], [w,u,v]
    belong to the same family of directions.

    u, v, w are first sorted and then a specific variant is selected where
    the order of u, v, w is 1, 0, 2 (i.e., w >= u >= v).
    """
    if isinstance(uvw, (list, tuple)):
        uvw = np.array(uvw)

    ndim = uvw.ndim
    if ndim == 1:
        uvw = uvw.reshape(1, -1)

    uvw = np.abs(uvw)
    # Sort u, v, w
    uvw = np.sort(uvw, axis=1)
    # Select variants where w >= u >= v
    uvw = uvw[:, [1, 0, 2]]

    R = uvw[:, 2] - uvw[:, 0]
    G = uvw[:, 0] - uvw[:, 1]
    B = uvw[:, 1]

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

    R = (R/kR)**pwr
    G = (G/kG)**pwr
    B = (B/kB)**pwr

    rgb = np.array([R, G, B])
    rgbmax = np.max(rgb, axis=0)
    # normalize rgb from 0 to 1 and then from 0 to 255
    rgb = rgb*255/rgbmax

    # rgb to int and invert axes (transpose)
    rgb = np.ndarray.astype(rgb, int).T

    if ndim == 1:
        rgb = rgb.ravel()

    return rgb


def unit_triangle(ax=None, n=512, **kwargs):
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


def plot_PF(R=None, M=None, proj=[1, 0, 0], ax=None,
            sel=None, parent_or=None, contour=False, verbose=True, **kwargs):
    """
    The user should provide either R or M. It's more convenient to use
    R values when plotting raw experimental data. M should be used 
    when plotting the variants of a specific orientation relationship.

    Parameters
    ----------
    R : numpy ndarray shape(N,3,3)
        Transformation matrix from the crystal coordinates to the 
        mechanical coordinates (EBSD system). Can be calculated 
        directly from the Euler angles provided by the EBSD system 
        using 'pyebsd.euler_rotation'
    M : numpy ndarray shape(N,3,3)
        Transformation matrix from the mechanical coordinates to the 
        crystal coordinates. M is the inverse (transposed) matrix of 
        R (M = R^-1)
    proj : list or numpy array(3) (optional)
        Family of direction projected in the pole figure.
        Default: [1,0,0]
    ax : AxesSubplot instance (optional)
        The pole figure will be plotted in the provided instance 'ax'
    sel : boolean numpy 1D array
        Array with boolean [True, False] values indicating which data 
        points should be plotted
        Default: None
    parent_or : numpy ndarray shape(3, 3)
        Orientation matrix of the parent phase. The pole figure is 
        rotated until the axes coincides with the orientation 
        'parent_or'
        Default: None
    contour : [True, False]
        contour=True plots the pole figure using contour plot
        Default: False

    **kwargs:
        lw_frame : float
            line width of PF frame
            Default: 0.5
        fill : [True, False]
            True: filled contour plot 'plt.contourf'; False: contour 
            plot 'plt.contour'
            Default: True
        bins : int or tuple or array (int,int)
            Binning used in the calculation of the points density 
            histogram (prior to contour plot)
            Default: (256, 256)
        fn : ['sqrt', 'log', 'None'] or function(x)
            function that modifies the points density.
            Default: 'sqrt'
        nlevels : int
            number of levels in the contour plot
            Default: 10

    The kwargs properties not listed here are automatically passed to 
    the plotting functions:
    if not contour:
        plt.plot(..., **kwargs)
    if contour and fill:
        plt.contour(..., **kwargs)
    if contour and not fill:
        plt.contourf(..., **kwargs)
    """
    if isinstance(R, np.ndarray):
        if R.ndim == 2:
            R = R.reshape(1, 3, 3)
    elif isinstance(M, np.ndarray):
        if M.ndim == 2:
            M = M.reshape(1, 3, 3)
        R = M.transpose([0, 2, 1])
    else:
        return

    if verbose:
        t0 = time.time()
        sys.stdout.write('Plotting Pole Figure... ')
        sys.stdout.flush()

    # gets Cartesian coordinates of PF projections
    xp, yp = PF(R, proj=proj, parent_or=parent_or)

    if isinstance(sel, np.ndarray):  # selected values
        xp, yp = xp[sel], yp[sel]

    if ax is None:  # if ax was not provided, creates new ax object
        fig, ax = plt.subplots(facecolor='white')
        ax.set_aspect('equal')
        ax.axis('off')
        lw_frame = kwargs.pop('lw_frame', .5)
        draw_circle_frame(ax, lw=lw_frame)

    if contour:
        fill = kwargs.pop('fill', True)
        bins = kwargs.pop('bins', (256, 256))

        hist, xedges, yedges = np.histogram2d(
            yp.ravel(), xp.ravel(), bins=bins, range=[[-1, 1], [-1, 1]])
        fn = kwargs.pop('fn', 'sqrt')

        if fn:
            if fn == 'sqrt':
                hist = hist**.5
            elif fn == 'log':
                hist = np.log(hist)
            else:
                try:
                    hist = fn(hist)
                except:
                    pass

        nlevels = kwargs.pop('nlevels', 10)
        lvls = np.linspace(0, np.max(hist), nlevels)
        kwargs['levels'] = lvls[1:]

        X, Y = np.meshgrid((xedges[:-1] + xedges[1:])/2.,
                           (yedges[:-1] + yedges[1:])/2.)
        circle = X**2 + Y**2 >= 1.
        hist[circle] = np.nan

        if fill:
            ax.contourf(hist, extent=(-1, 1, -1, 1), **kwargs)
        else:
            ax.contour(hist, extent=(-1, 1, -1, 1), **kwargs)
    else:
        if not kwargs.get('marker', None):
            kwargs['marker'] = '.'
        if not kwargs.get('markersize', None) and not kwargs.get('ms', None):
            kwargs['markersize'] = 1
        ax.plot(xp.ravel(), yp.ravel(), linestyle='None', **kwargs)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    if verbose:
        sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))

    return ax


def plot_property(prop, nrows, ncols_even, ncols_odd, x, y,
                  dx=None, ax=None, colordict=None, colorfill=[0, 0, 0, 1],
                  sel=None, gray=None, tiling='rect', w=2048, scalebar=True,
                  verbose=True, **kwargs):
    """
    Documentation
    """
    if verbose:
        t0 = time.time()
        sys.stdout.write('Plotting property map... ')
        sys.stdout.flush()

    # getting kwargs properties...
    cmap = kwargs.pop('cmap', plt.get_cmap())
    vmin, vmax = kwargs.pop('vmin', np.min(prop[sel])), kwargs.pop(
        'vmax', np.max(prop[sel]))

    # calculates expect number of points from provided nrows,
    # ncols_odd and ncols_even
    N = int((nrows//2)*(ncols_odd + ncols_even) + (nrows % 2)*ncols_odd)
    xmin, xmax = np.min(x[sel]), np.max(x[sel])
    ymin, ymax = np.min(y[sel]), np.max(y[sel])

    if isinstance(colordict, dict):
        col = np.ndarray((prop.shape[0], 4))
        col[:] = colorfill
        for p, rgba in colordict.items():
            col[prop == float(p)] = rgba
    else:
        prop = (prop - vmin)/(vmax - vmin)  # normalizes prop to range [0,1]
        prop[prop < 0], prop[prop > 1] = 0, 1
        col = cmap(prop)

    if N != prop.shape[0]:
        return

    if isinstance(sel, np.ndarray):
        if prop.shape != sel.shape:
            return
        else:
            col[np.logical_not(sel)] = colorfill
    else:
        sel = np.ndarray(N, dtype=bool)
        sel.fill(True)

    if isinstance(gray, np.ndarray):
        if prop.shape != gray.shape:
            return
        else:
            gray = gray.reshape(-1, 1)/np.max(gray)
            col[sel] = col[sel]*gray[sel]
            col[:, 3] = 1.  # set alpha = 1. for all points

    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.cla()
        fig = ax.get_figure()

    scalebar_location = kwargs.pop('scalebar_location', 'lower left')

    if tiling == 'hex':
        if not dx:
            dx = (np.max(x) - np.min(x))/ncols_even

        col = (255*col[sel]).astype(int)
        x_hex = np.ndarray((len(x[sel]), 6))
        y_hex = np.ndarray((len(y[sel]), 6))
        edge_length = dx/3.**.5

        for i in range(6):
            x_hex[:, i] = x[sel] + np.sin(i*np.pi/3)*edge_length
            y_hex[:, i] = y[sel] + np.cos(i*np.pi/3)*edge_length

        scale = 1.*w/(xmax - xmin)
        h = np.int(scale*(ymax - ymin))

        x_hex = (x_hex - xmin)*scale
        y_hex = (y_hex - ymin)*scale

        img = Image.new('RGB', (w, h), 'black')
        draw = ImageDraw.Draw(img)
        for i in range(len(x_hex)):
            color = col[i, 0], col[i, 1], col[i, 2]
            hexagon = list(zip(*[x_hex[i], y_hex[i]]))
            draw.polygon(hexagon, fill=color)

        ax.imshow(img, interpolation='nearest', extent=(
            xmin, xmax, ymax, ymin), **kwargs)
    elif tiling == 'rect':
        N, ncols = 2*N, 2*ncols_even
        # remove extra pixels
        rm = np.hstack([np.arange(0, N, 2*(ncols+1)),
                        np.arange(ncols+1, N, 2*(ncols+1))])

        col = np.repeat(col, 2, axis=0)
        col = np.delete(col, rm, axis=0)

        imin, imax = 0, nrows
        jmin, jmax = 0, ncols
        if np.count_nonzero(np.logical_not(sel)) > 0:
            sel = np.repeat(sel, 2)
            sel = np.delete(sel, rm, axis=0)
            isel, jsel = np.where(sel.reshape(nrows, ncols))
            imin, imax = np.min(isel)+1, np.max(isel)+1
            jmin, jmax = np.min(jsel)+1, np.max(jsel)

        scale = 1.*w/(jmax - jmin)
        h = np.int(scale*(imax - imin)*(3.**.5))

        col = col.reshape(nrows, ncols, -1)

        img = toimage(col[imin:imax, jmin:jmax, :])
        img = img.resize(size=(w, h))
        ax.imshow(img, interpolation='nearest', extent=(
            xmin, xmax, ymax, ymin), **kwargs)
    else:
        return

    # add scalebar
    if scalebar:
        scalebar = ScaleBar(1e-6)
        scalebar.location = scalebar_location
        ax.add_artist(scalebar)

    # removing the borders/margins
    ax.axis('off')
    set_tight_plt()

    if verbose:
        sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))

    # return ax, img
    return EBSDMap(x, y, img, ax, fig)


def plot_IPF(R, nrows, ncols_even, ncols_odd, x, y,
             dx=None, d='ND', ax=None, sel=None, gray=None, tiling='rect',
             w=2048, scalebar=True, verbose=True, **kwargs):
    """
    Documentation
    """
    if verbose:
        t0 = time.time()
        sys.stdout.write('Plotting Inverse Pole Figure... ')
        sys.stdout.flush()

    # calculates expect number of points from provided nrows,
    # ncols_odd and ncols_even
    N = int((nrows//2)*(ncols_odd + ncols_even) + (nrows % 2)*ncols_odd)
    xmin, xmax = np.min(x[sel]), np.max(x[sel])
    ymin, ymax = np.min(y[sel]), np.max(y[sel])
    # call IPF to get crystal directions parallel to d and
    # convert to color code (RGB)
    col = get_color_IPF(IPF(R, d))

    if N != R.shape[0]:
        print('N and R.shape differ')
        return

    if isinstance(sel, np.ndarray):
        if N != sel.shape[0]:
            print('R.shape and sel.shape differ')
            return
        else:
            col[np.logical_not(sel)] = [0, 0, 0]  # RGB
    else:
        sel = np.ndarray(N, dtype=bool)
        sel.fill(True)

    if isinstance(gray, np.ndarray):
        if N != gray.shape[0]:
            print('R.shape and gray.shape differ')
            return
        else:
            gray = gray.reshape(-1, 1)/np.max(gray)
            col[sel] = col[sel]*gray[sel]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.cla()
        fig = ax.get_figure()

    scalebar_location = kwargs.pop('scalebar_location', 'lower left')

    if tiling == 'hex':
        if not dx:
            dx = (np.max(x) - np.min(x))/ncols_even

        col = col[sel]
        x_hex = np.ndarray((len(x[sel]), 6))
        y_hex = np.ndarray((len(y[sel]), 6))
        edge_length = dx/3.**.5

        for i in range(6):
            # coordinates of the vertices of each hexagonal tile
            # in physical units (most commonly nm)
            x_hex[:, i] = x[sel] + np.sin(i*np.pi/3)*edge_length
            y_hex[:, i] = y[sel] + np.cos(i*np.pi/3)*edge_length

        scale = 1.*w/(xmax - xmin)
        h = np.int((ymax - ymin)*scale)

        # coordinates of the vertices in pixels
        x_hex = (x_hex - xmin)*scale
        y_hex = (y_hex - ymin)*scale

        img = Image.new('RGB', (w, h), 'black')
        draw = ImageDraw.Draw(img)
        for i in range(len(x_hex)):
            color = col[i, 0], col[i, 1], col[i, 2]
            hexagon = list(zip(*[x_hex[i], y_hex[i]]))
            draw.polygon(hexagon, fill=color)

        ax.imshow(img, interpolation='nearest', extent=(
            xmin, xmax, ymax, ymin), **kwargs)
    elif tiling == 'rect':
        N, ncols = 2*N, 2*ncols_even
        # remove extra pixels
        rm = np.hstack([np.arange(0, N, 2*(ncols+1)),
                        np.arange(ncols+1, N, 2*(ncols+1))])

        col = np.repeat(col, 2, axis=0)
        col = np.delete(col, rm, axis=0)

        imin, imax = 0, nrows
        jmin, jmax = 0, ncols
        if np.count_nonzero(np.logical_not(sel)) > 0:
            sel = np.repeat(sel, 2)
            sel = np.delete(sel, rm, axis=0)
            isel, jsel = np.where(sel.reshape(nrows, ncols))
            imin, imax = np.min(isel)+1, np.max(isel)+1
            jmin, jmax = np.min(jsel)+1, np.max(jsel)

        scale = 1.*w/(jmax - jmin)
        h = np.int(scale*(imax - imin)*(3.**.5))

        col = col.reshape(nrows, ncols, -1)

        img = toimage(col[imin:imax, jmin:jmax, :])
        img = img.resize(size=(w, h))
        ax.imshow(img, interpolation='nearest', extent=(
            xmin, xmax, ymax, ymin), **kwargs)
    else:
        return

    # add scalebar
    if scalebar:
        scalebar = ScaleBar(1e-6)
        scalebar.location = scalebar_location
        ax.add_artist(scalebar)

    # removing the borders/margins
    ax.axis('off')
    set_tight_plt()

    if verbose:
        sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))

    # return ax, img
    return EBSDMap(x, y, img, ax, fig)
