import sys
import os
import time

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from itertools import permutations

from .orientation import euler_angles_to_rotation_matrix, PF, IPF, stereographic_projection
from ..draw import modify_show, set_tight_plt, draw_circle_frame, toimage, ScaleBar
from ..selection import LassoSelector2, RectangleSelector2

__THRESHOLD_TILING__ = 10000


def set_threshold_tiling(threshold):
    """
    Sets __THRESHOLD_TILING__

    Parameters
    ----------
    threshold : int
        New value of __THRESHOLD_TILING__
    """
    global __THRESHOLD_TILING__
    __THRESHOLD_TILING__ = int(threshold)


class _CoordsFormatter(object):
    def __init__(self, extent, Z):
        self.xmin = min(extent[:2])
        self.ymin = min(extent[2:])
        self.xrng = abs(extent[1] - extent[0])
        self.yrng = abs(extent[3] - extent[2])
        self.Z = Z
        self.h, self.w = self.Z.shape

    def __call__(self, x, y):
        i = int(self.w*(x - self.xmin)/self.xrng)
        j = int(self.h*(y - self.ymin)/self.yrng)
        i = self.w - 1 if i == self.w else i
        j = self.h - 1 if j == self.h else j
        return 'x={:g}    y={:g}    z={:g}'.format(x, y, self.Z[j, i])


class EBSDMap(object):
    """
    Documentation
    """

    def __init__(self, x, y, img, ax, fig, cax=None):
        self.x = x
        self.y = y
        self.img = img
        self.ax = ax
        self.fig = fig
        self.cax = cax

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
        self.selector = LassoSelector2(self.ax, self.x, self.y, lineprops=lineprops)
        return self.selector

    def rect_selector(self, rectprops=dict(edgecolor='white', fill=False), aspect=None):
        self.selector = RectangleSelector2(self.ax, self.x, self.y, rectprops=rectprops, aspect=aspect)
        return self.selector


def _calculate_barycenter_unit_triangle():
    from ..crystal import stereographic_projection_to_direction

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

    return stereographic_projection_to_direction([Cx, Cy])


def get_color_IPF(uvw, **kwargs):
    """
    Get the IPF color(s) of a given uvw direction or list of directions.

    So far it is only implemented for the cubic system.
    It is first necessary to find the direction (in the family of directions)
    that falls inside the unit triangle. Instead of calculating all directions
    using the 24 symmetry operators, this function explores some properties
    of the cubic system. 
    In order to a given uvw direction fall inside the unit triangle (delimited
    by the directions 001, 101, and 111), it suffices that u, v, and w are all
    positive numbers and w >= u >= v.
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

    img_pil = toimage(col.reshape(n, n, 3))

    if ax is None:
        fig, ax = plt.subplots(facecolor='white')

    ax.set_aspect('equal')
    ax.axis('off')
    img = ax.imshow(img_pil, interpolation='None', origin='lower', extent=(0, xmax, 0, ymax))

    # Draw borders of unit triangle
    t = np.linspace(0, 1., n)

    # [np.repeat(1.,n), t, np.repeat(1.,n)]
    # [t[::-1], t[::-1], np.repeat(1.,n)]
    # [t, np.repeat(0,n), np.repeat(1.,n)]

    # u = np.hstack([np.repeat(1., n), t[::-1], t])
    # v = np.hstack([t, t[::-1], np.repeat(0, n)])
    # w = np.hstack([np.repeat(1., n), np.repeat(1., n), np.repeat(1., n)])
    uvw = np.full((3*n, 3), 1.)
    # u
    uvw[n:2*n, 0] = t[::-1]
    uvw[2*n:, 0] = t
    # v
    uvw[:n, 1] = t
    uvw[n:2*n, 1] = t[::-1]
    uvw[2*n:, 1] = 0.
    # w: Nothing do to. All values are equal to 1.

    x, y = stereographic_projection(uvw)
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


def plot_PF(M=None, proj=[1, 0, 0], ax=None, sel=None, rotation=None, contour=False,
            verbose=True, **kwargs):
    """
    The user should provide either R or M. It's more convenient to use
    R values when plotting raw experimental data. M should be used 
    when plotting the variants of a specific orientation relationship.

    Parameters
    ----------
    M : numpy ndarray shape(N,3,3)
        Transformation matrix from the sample coordinate frame to
        the crystal coordinate frame.
    proj : list or numpy array(3) (optional)
        Family of direction projected in the pole figure.
        Default: [1,0,0]
    ax : AxesSubplot instance (optional)
        The pole figure will be plotted in the provided instance 'ax'
    sel : boolean numpy 1D array
        Array with boolean [True, False] values indicating which data 
        points should be plotted
        Default: None
    rotation : list or array shape(3,3)
        Rotation matrix that rotates the pole figure.
        The columns of the matrix correspond to the directions parallel to 
        the axes of the pole figure.
        Default: None
    contour : [True, False]
        contour=True plots the pole figure using contour plot
        Default: False

    **kwargs:
        R : numpy ndarray shape(N,3,3)
            If M is not provided, R has to be provided instead.
            Transformation matrix from the crystal coordinate frame to 
            the sample coordinate frame (EBSD system). 
            R is the inverse/transposed matrix of M (R = M^-1) 
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
    R = kwargs.pop('R', None)

    if isinstance(R, np.ndarray):
        if R.ndim == 2:
            R = R.reshape(1, 3, 3)
    elif isinstance(M, np.ndarray):
        if M.ndim == 2:
            M = M.reshape(1, 3, 3)
        R = M.transpose([0, 2, 1])
    else:
        raise Exception('M or R has to be provided')

    if verbose:
        t0 = time.time()
        sys.stdout.write('Plotting Pole Figure... ')
        sys.stdout.flush()

    # PF returns directions (in the sample coordinate frame) of all variants
    # of the crytal direction proj
    # dsample has shape shape(N, nvar, 3), where nvar is the number of
    # variants of proj
    dsample = PF(R, proj=proj, rotation=rotation)

    if isinstance(sel, np.ndarray):  # selected values
        dsample = dsample[sel]

    # flattens dsample along the axes 0 and 1 (N and nvar).
    dsample = dsample.reshape(-1, 3)

    # calculate the Cartensian coordinates of the stereographic projection
    # for only the directions where the z coordinate is larger or equal
    # than 0
    xp, yp = stereographic_projection(dsample[dsample[:, 2] >= 0])

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
        if kwargs.pop('scatter', False):
            ax.scatter(xp.ravel(), yp.ravel(), **kwargs)
        else:
            if not kwargs.get('linestyle', None):
                kwargs['linestyle'] = 'None'
            if not kwargs.get('marker', None):
                kwargs['marker'] = '.'
            if not kwargs.get('markersize', None) and not kwargs.get('ms', None):
                kwargs['markersize'] = 1

            ax.plot(xp.ravel(), yp.ravel(), **kwargs)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    if verbose:
        sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))

    return ax


def plot_property(prop, nrows, ncols_even, ncols_odd, x, y,
                  dx=None, ax=None, colordict=None, colorfill=[0, 0, 0, 1],
                  sel=None, gray=None, grid='HexGrid', tiling=None,
                  w=2048, scalebar=True, colorbar=True, verbose=True, **kwargs):
    """
    Documentation
    """
    # get N based on nrows, ncols_odd and ncols_even
    if verbose:
        t0 = time.time()
        sys.stdout.write('Plotting property map... ')
        sys.stdout.flush()

    # get N based on nrows, ncols_odd and ncols_even
    if grid.lower() == 'hexgrid':
        N = int((nrows//2)*(ncols_odd + ncols_even) + (nrows % 2)*ncols_odd)
    elif grid.lower() == 'sqrgrid':
        N = nrows*ncols_odd
    else:
        raise Exception('Unknown grid type "{}"'.format(grid))

    if N != len(prop):
        raise Exception('N and len(prop) differ')

    if isinstance(sel, np.ndarray):
        if N != len(sel):
            raise Exception('N and len(sel) differ')
        sel &= ~np.isnan(prop)
    else:
        sel = ~np.isnan(prop)
    not_sel = ~sel

    # set default tiling
    if grid.lower() == 'sqrgrid' and tiling == 'hex':
        print('hex tiling not supported for squared grid. Using rect tiling instead.')
        tiling = 'rect'

    if tiling is None:
        tiling = 'rect'
        if grid.lower() == 'hexgrid':
            if np.count_nonzero(sel) <= __THRESHOLD_TILING__:
                tiling = 'hex'

    # x and y plot limits
    xmin, xmax = np.min(x[sel]), np.max(x[sel])
    ymin, ymax = np.min(y[sel]), np.max(y[sel])

    if dx is None:
        dx = (np.max(x) - np.min(x))/ncols_odd

    ymin -= dx/2.
    ymax += dx/2.
    if grid.lower() == 'sqrgrid':
        xmin -= dx/2.
        xmax += dx/2.

    # getting kwargs parameters
    cmap = kwargs.pop('cmap', plt.get_cmap())
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    scalebar_location = kwargs.pop('scalebar_location', 'lower left')
    vmin = kwargs.pop('vmin', np.min(prop[sel]))
    vmax = kwargs.pop('vmax', np.max(prop[sel]))

    # coloring
    col = np.ndarray((N, 4))
    col[:] = colorfill
    if isinstance(colordict, dict):
        for p, rgba in colordict.items():
            col[prop == float(p)] = rgba
    else:
        # normalizes prop to range [0,1] and makes colormap
        col[sel] = cmap(((prop - vmin)/(vmax - vmin))[sel])

    # filling invalid/non-selected data points
    col[not_sel] = colorfill
    if prop.dtype == float:
        prop[not_sel] = np.nan

    # applying gray mask
    if isinstance(gray, np.ndarray):
        if N != gray.shape[0]:
            raise Exception('M.shape and gray.shape differ')
        else:
            gray = gray.reshape(-1, 1)/np.max(gray)
            col[sel] = col[sel]*gray[sel]
            col[:, 3] = 1.  # set alpha = 1. for all points

    # getting AxesSubplot object
    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.cla()
        fig = ax.get_figure()

    # plotting maps
    if tiling == 'hex':
        col = (255*col[sel]).astype(int)
        x_hex = np.ndarray((len(x[sel]), 6))
        y_hex = np.ndarray((len(y[sel]), 6))
        edge_length = dx/3.**.5

        # calculates the coordinates of the vertices of the hexagonal pixel
        for i in range(6):
            x_hex[:, i] = x[sel] + np.sin(i*np.pi/3)*edge_length
            y_hex[:, i] = y[sel] + np.cos(i*np.pi/3)*edge_length

        scale = 1.*w/(xmax - xmin)
        h = np.int(scale*(ymax - ymin))

        x_hex = (x_hex - xmin)*scale
        y_hex = (y_hex - ymin)*scale

        img_pil = Image.new('RGB', (w, h), 'black')
        draw = ImageDraw.Draw(img_pil)
        for i in range(len(x_hex)):
            color = col[i, 0], col[i, 1], col[i, 2]
            hexagon = list(zip(*[x_hex[i], y_hex[i]]))
            draw.polygon(hexagon, fill=color)

    elif tiling == 'rect':
        if grid.lower() == 'hexgrid':
            N, ncols = 2*N, 2*ncols_even
            # double pixels
            sel = np.repeat(sel, 2)
            col = np.repeat(col, 2, axis=0)
            prop = np.repeat(prop, 2, axis=0)
            # remove extra pixels
            rm = np.hstack([np.arange(0, N, 2*(ncols+1)),
                            np.arange(ncols+1, N, 2*(ncols+1))])
            sel = np.delete(sel, rm, axis=0)
            col = np.delete(col, rm, axis=0)
            prop = np.delete(prop, rm, axis=0)
        else:  # sqrgrid
            ncols = ncols_odd

        isel, jsel = np.where(sel.reshape(nrows, ncols))
        jmin, jmax = np.min(jsel), np.max(jsel) + 1  # x
        imin, imax = np.min(isel), np.max(isel) + 1  # y

        # crop first or last column
        if grid.lower() == 'hexgrid':
            if jmin != 0 and jmin != ncols:
                jmin += 1
            if jmax != 0 and jmax != ncols:
                jmax -= 1

        scale = 1.*w/(xmax - xmin)
        if grid.lower() == 'hexgrid':
            h = np.int(scale*(ymax - ymin)*(3.**.5))
        else:
            h = np.int(scale*(ymax - ymin))

        col = col.reshape(nrows, ncols, -1)
        prop = prop.reshape(nrows, ncols)

        img_pil = toimage(col[imin:imax, jmin:jmax, :])
        img_pil = img_pil.resize(size=(w, h))
        ax.format_coord = _CoordsFormatter((xmin, xmax, ymax, ymin), prop[imin:imax, jmin:jmax])

    else:
        plt.close(fig)
        raise Exception('Unknown "{}" tiling'.format(tiling))

    img = ax.imshow(img_pil, interpolation='None', extent=(xmin, xmax, ymax, ymin), **kwargs)

    # add scalebar
    if scalebar:
        scalebar = ScaleBar(1e-6)
        scalebar.location = scalebar_location
        ax.add_artist(scalebar)

    # add colorbar
    cax = None
    if colorbar and colordict is None:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.colorbar(sm, ax=ax, shrink=.92)

    # removing the borders/margins
    ax.axis('off')
    set_tight_plt(fig, ax)

    if verbose:
        sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))

    return EBSDMap(x, y, img, ax, fig, cax)


def plot_IPF(M, nrows, ncols_even, ncols_odd, x, y,
             dx=None, d=[0, 0, 1], ax=None, sel=None, gray=None, grid='HexGrid',
             tiling=None, w=2048, scalebar=True, verbose=True, **kwargs):
    """
    Documentation
    """
    if verbose:
        t0 = time.time()
        sys.stdout.write('Plotting Inverse Pole Figure... ')
        sys.stdout.flush()

    # get N based on nrows, ncols_odd and ncols_even
    if grid.lower() == 'hexgrid':
        N = int((nrows//2)*(ncols_odd + ncols_even) + (nrows % 2)*ncols_odd)
    elif grid.lower() == 'sqrgrid':
        N = nrows*ncols_odd
    else:
        raise Exception('Unknown grid type "{}"'.format(grid))

    if N != len(M):
        raise Exception('N and len(M) differ')

    if isinstance(sel, np.ndarray):
        if N != sel.shape[0]:
            raise Exception('N and len(sel) differ')
    else:
        sel = np.full(N, True)
    not_sel = ~sel

    # set default tiling
    if grid.lower() == 'sqrgrid' and tiling == 'hex':
        print('hex tiling not supported for squared grid. Using rect tiling instead.')
        tiling = 'rect'

    if tiling is None:
        tiling = 'rect'
        if grid.lower() == 'hexgrid':
            if np.count_nonzero(sel) <= __THRESHOLD_TILING__:
                tiling = 'hex'

    # x and y plot limits
    xmin, xmax = np.min(x[sel]), np.max(x[sel])
    ymin, ymax = np.min(y[sel]), np.max(y[sel])

    if dx is None:
        dx = (np.max(x) - np.min(x))/ncols_odd

    ymin -= dx/2.
    ymax += dx/2.
    if grid.lower() == 'sqrgrid':
        xmin -= dx/2.
        xmax += dx/2.

    # getting kwargs parameters
    scalebar_location = kwargs.pop('scalebar_location', 'lower left')

    # call IPF to get crystal directions parallel to d and
    # convert to color code (RGB)
    col = get_color_IPF(IPF(M, d))
    # filling invalid/non-selected data points
    col[not_sel] = [0, 0, 0]  # RGB

    # applying gray mask
    if isinstance(gray, np.ndarray):
        if N != gray.shape[0]:
            raise Exception('N and len(gray) differ')
        else:
            gray = gray.reshape(-1, 1)/np.max(gray)
            col[sel] = col[sel]*gray[sel]

    # getting AxesSubplot object
    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.cla()
        fig = ax.get_figure()

    # plotting maps
    if tiling == 'hex':
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

        img_pil = Image.new('RGB', (w, h), 'black')
        draw = ImageDraw.Draw(img_pil)
        for i in range(len(x_hex)):
            color = col[i, 0], col[i, 1], col[i, 2]
            hexagon = list(zip(*[x_hex[i], y_hex[i]]))
            draw.polygon(hexagon, fill=color)

    elif tiling == 'rect':
        if grid.lower() == 'hexgrid':
            N, ncols = 2*N, 2*ncols_even
            # double pixels
            sel = np.repeat(sel, 2)
            col = np.repeat(col, 2, axis=0)
            # remove extra pixels
            rm = np.hstack([np.arange(0, N, 2*(ncols+1)),
                            np.arange(ncols+1, N, 2*(ncols+1))])
            sel = np.delete(sel, rm, axis=0)
            col = np.delete(col, rm, axis=0)
        else:  # sqrgrid
            ncols = ncols_odd

        isel, jsel = np.where(sel.reshape(nrows, ncols))
        jmin, jmax = np.min(jsel), np.max(jsel) + 1  # x
        imin, imax = np.min(isel), np.max(isel) + 1  # y

        # crop first or last column
        if grid.lower() == 'hexgrid':
            if jmin != 0 and jmin != ncols:
                jmin += 1
            if jmax != 0 and jmax != ncols:
                jmax -= 1

        scale = 1.*w/(xmax - xmin)
        if grid.lower() == 'hexgrid':
            h = np.int(scale*(ymax - ymin)*(3.**.5))
        else:
            h = np.int(scale*(ymax - ymin))

        col = col.reshape(nrows, ncols, -1)

        img_pil = toimage(col[imin:imax, jmin:jmax, :])
        img_pil = img_pil.resize(size=(w, h))

    else:
        plt.close(fig)
        raise Exception('Unknown "{}" tiling'.format(tiling))

    img = ax.imshow(img_pil, interpolation='None', extent=(xmin, xmax, ymax, ymin), **kwargs)

    # add scalebar
    if scalebar:
        scalebar = ScaleBar(1e-6)
        scalebar.location = scalebar_location
        ax.add_artist(scalebar)

    # removing the borders/margins
    ax.axis('off')
    set_tight_plt(fig, ax)

    if verbose:
        sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))

    return EBSDMap(x, y, img, ax, fig)
