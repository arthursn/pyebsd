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

__all__ = ['set_threshold_tiling', 'EBSDMap', 'get_color_IPF',
           'unit_triangle', 'plot_PF', 'plot_property', 'plot_IPF']

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
    """
    Formats coordinates and z values in interactive plot mode
    """

    def __init__(self, extent, Z, name='z'):
        self.xmin = min(extent[:2])
        self.ymin = min(extent[2:])
        self.xrng = abs(extent[1] - extent[0])
        self.yrng = abs(extent[3] - extent[2])
        self.Z = Z
        self.name = name
        self.jmax = self.Z.shape[1] - 1
        self.imax = self.Z.shape[0] - 1

    def __call__(self, x, y):
        j = int(round(self.jmax*(x - self.xmin)/self.xrng))
        i = int(round(self.imax*(y - self.ymin)/self.yrng))
        string = 'x={:g}    y={:g}'.format(x, y)
        if i > 0 and j > 0:
            try:
                if not np.any(np.isnan(self.Z[i, j])):
                    string += '    {}={}'.format(self.name, self.Z[i, j])
            except:
                pass
        return string


class EBSDMap(object):
    """
    Stores EBSD map plotting information (img, ax, fig, cax) and
    provides wrapper for selector functions

    Parameters
    ----------
    x : numpy ndarray
        x pixel coordinates
    y : numpy ndarray
        y pixel coordinates
    img : matplotlib AxesImage object
        AxesImage object
    ax : matplotlib AxesSubplot object
        AxesSubplot object
    fig : matplotlib Figure object
        Figure object
    cax : matplotlib Colorbar object (optional)
        Colorbar object
        Default: None
    """

    def __init__(self, x, y, img, ax, fig, cax=None):
        self.x = x
        self.y = y
        self.img = img
        self.ax = ax
        self.fig = fig
        self.fig.canvas.mpl_connect('draw_event', self.ondraw)
        self.cax = cax
        self.xlim = None
        self.ylim = None
        self._selector = None

    @property
    def sel(self):
        """
        Boolean numpy ndarray masking pixels inside selection
        """
        if self.selector is not None:
            return self.selector.sel

    @property
    def selector(self):
        """
        Selector object, either LassoSelector2 or RectangleSelector2
        """
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

    def ondraw(self, event):
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()

    def get_xlim(self):
        return self.xlim

    def get_ylim(self):
        return self.ylim

    def lasso_selector(self, lineprops=dict(color='white')):
        """
        Initializes LassoSelector2
        """
        self.selector = LassoSelector2(self.ax, self.x, self.y, lineprops=lineprops)
        return self.selector

    def rect_selector(self, rectprops=dict(edgecolor='white', fill=False), aspect=None):
        """
        Initializes RectangleSelector2
        """
        self.selector = RectangleSelector2(self.ax, self.x, self.y, rectprops=rectprops, aspect=aspect)
        return self.selector


def _calculate_barycenter_unit_triangle():
    """
    Function name speaks for itself
    """
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

    shape = uvw.shape  # original shape
    ndim = uvw.ndim  # original number of dimensions
    if ndim != 2:
        uvw = uvw.reshape(-1, 3)

    if not kwargs.pop('issorted', False):
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
    rgb = rgb.astype(np.uint8).T

    if ndim != 2:
        rgb = rgb.reshape(shape)  # reshapes to match original shape of uvw

    return rgb


def unit_triangle(ax=None, n=512, **kwargs):
    """
    Creates IPF unit triangle. Only valid for cubic system

    Parameters
    ----------
    ax : matplotlib AxesSubplot object
        AxesSubplot object
    n : int (optional)
        Rasterization resolution
        Default: 512 

    **kwargs :
        kwargs parameters are passed to get_color_IPF function
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

    color = np.ndarray(uvw.shape)
    # select directions that will fit inside the unit triangle, i.e.,
    # only those where w >= u >= v
    sel = (w >= u) & (u >= v)
    # uvw directions to corresponding color
    color[sel] = get_color_IPF(uvw[sel], **kwargs)
    # fill points outside the unit triangle in white
    color[~sel] = [255, 255, 255]

    img_pil = toimage(color.reshape(n, n, 3))

    if ax is None:
        fig, ax = plt.subplots(facecolor='white')

    ax.set_aspect('equal')
    ax.axis('off')
    # Plots the unit triangle
    img = ax.imshow(img_pil, interpolation='None', origin='lower', extent=(0, xmax, 0, ymax))

    # Coordinates displayed in the interactive plot window
    _uvw = uvw.copy()
    _uvw[~sel] = [np.nan, np.nan, np.nan]
    ax.format_coord = _CoordsFormatter((0, xmax, 0, ymax), _uvw.reshape(n, n, 3).round(6), 'd')

    # Calculate coordinates of borders of unit triangle
    t = np.linspace(0, 1., n)
    # Trust me, this works
    uvw = np.full((3*n, 3), 1., dtype=float)
    # u
    uvw[n:2*n, 0] = t[::-1]
    uvw[2*n:, 0] = t
    # v
    uvw[:n, 1] = t
    uvw[n:2*n, 1] = t[::-1]
    uvw[2*n:, 1] = 0.
    # w: Nothing do to. All values are equal to 1

    x, y = stereographic_projection(uvw)
    # Plots borders of unit triangle
    ax.plot(x, y, 'k-', lw=2)

    # Reference directions
    ax.annotate('001', xy=stereographic_projection([0, 0, 1]), xytext=(0, -10),
                textcoords='offset points', ha='center', va='top', size=30)
    ax.annotate('101', xy=stereographic_projection([1, 0, 1]), xytext=(0, -10),
                textcoords='offset points', ha='center', va='top', size=30)
    ax.annotate('111', xy=stereographic_projection([1, 1, 1]), xytext=(0, 10),
                textcoords='offset points', ha='center', va='bottom', size=30)
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
        Default: [1, 0, 0]
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

        hist, xedges, yedges = np.histogram2d(yp.ravel(), xp.ravel(), bins=bins, range=[[-1, 1], [-1, 1]])
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


def plot_property(prop, nrows, ncols_odd, ncols_even, x, y, dx=None, dy=None,
                  ax=None, colordict=None, colorfill='black', fillvalue=np.nan,
                  sel=None, gray=None, grid='HexGrid', tiling=None, w=2048,
                  scalebar=True, colorbar=True, verbose=True, **kwargs):
    """
    Plots any EBSD property

    Parameters
    ----------
    prop : array shape(N)
        Property to be plotted provided as np.ndarray(N), where N is the
        size of the data file
    nrows : int
        Number of rows
    ncols_odd : int
        Number of columns in the odd rows
    ncols_even : int
        Number of columns in the even rows
    x : numpy ndarray shape(N)
        x pixel coordinates
    y : numpy ndarray shape(N)
        y pixel coordinates
    dx : float (optional)
        Grid spacing along x coordinates. If None is provided, guesses
        it from x.
        Default: None
    dy : float (optional)
        Grid spacing along y coordinates. If None is provided, guesses
        it from y.
        Default: None
    ax : AxesSubplot object (optional)
        The pole figure will be plotted in the provided object 'ax'
        Default: None
    colordict : dict(str: str or list) (optional)
        Dictionary that maps indexed phase to respective color provided
        as string, list shape(3) (RGB), or list shape(4) (RGBA)
        E.g: {'1': 'red', '2': 'green'}
        If None is provided, the the colors are assigned automatically
        by cycling through the classic matplotlib colors defined in 
        the data member colors (self.colors, which can be changed 
        at will). Colors are assigned to the phases in alphabetical 
        (numerical) order
        Default: None
    colorfill : str or list shape(3) or shape(4) (optional)
        Color used to fill unindexed pixels. It can be provided as RGB 
        or RGBA values as an iterable. If RGBA is provided, alpha channel
        is droppped
        Default: 'black'
    fillvalue : float, int
        Value used to fill non valid/non selected points 
        Default: np.nan
    sel : bool numpy 1D array (optional)
        Boolean array indicating which data points should be plotted
        Default: None
    gray : numpy ndarray (optional)
        Grayscale mask plotted over IPF.
        For example, one may want to overlay the IPF map with the image
        quality data.
        Default: None
    grid : str (optional)
        Grid type
        Default: 'HexGrid'
    tiling : str (optional)
        Valid options are 'rect' or 'hex'
        If no option is provided, uses as default 'rect' if 
        N > __THRESHOLD_TILING__, else 'hex'. By default, the value of
        __THRESHOLD_TILING__ is 10000, but it can be set to any value
        by calling pyebsd.set_threshold_tiling(..)
        Default: None
    w : int (optional)
        Width in pixel
        Default: 2048
    scalebar : bool (optional)
        If True, displays scalebar over IPF map
        Default: True
    verbose : bool (optional)
        If True, prints computation time
        Default: True

    **kwargs :
        kwargs parameters are passed to function ax.imshow:
        ax.imshow(img, ..., **kwargs)

    Returns
    -------
    ebsdmap : EBSDMap object
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
    if dy is None:
        dy = (np.max(y) - np.min(y))/(nrows - 1)

    if tiling == 'hex':
        edge_length = dx/3.**.5
        ymin -= edge_length/2.
        ymax += edge_length/2.
    else:
        ymin -= dy/2.
        ymax += dy/2.

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

    # min and max values of the gray mask
    graymin = kwargs.pop('graymin', None)
    graymax = kwargs.pop('graymax', None)

    # converts string, or list to tuple with RGB color. Drops alpha channel if RGBA is provided
    colorfill = matplotlib.colors.to_rgb(colorfill)

    # coloring
    color = np.ndarray((N, 3))
    color[:] = colorfill
    if isinstance(colordict, dict):
        for p, color_code in colordict.items():
            color[prop == float(p)] = matplotlib.colors.to_rgb(color_code)
    else:
        # normalizes prop to range [0,1] and makes rgb colormap
        color[sel] = cmap(((prop - vmin)/(vmax - vmin))[sel])[:, :3]

    # makes copy of prop
    _prop = prop.copy()
    # filling invalid/non-selected data points
    color[not_sel] = colorfill
    if fillvalue is np.nan:
        _prop = _prop.astype(float)
    _prop[not_sel] = fillvalue

    # applying gray mask
    if isinstance(gray, np.ndarray):
        if N != gray.shape[0]:
            raise Exception('M.shape and gray.shape differ')
        else:
            if graymin is None:
                graymin = gray.min()
            if graymax is None:
                graymax = gray.max()
            gray = (gray.reshape(-1, 1) - graymin)/(graymax - graymin)
            gray[gray < 0.] = 0.
            gray[gray > 1.] = 1.
            color[sel] = color[sel]*gray[sel]

    # getting AxesSubplot object
    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.cla()
        fig = ax.get_figure()

    if grid.lower() == 'hexgrid':
        N, ncols = 2*N, 2*min(ncols_odd, ncols_even)  # N pixels and ncols for rect grid plotting
        if ncols_odd > ncols_even:
            rm = np.hstack([np.arange(0, N, 2*(ncols+1)),
                            np.arange(ncols+1, N, 2*(ncols+1))])
        else:
            rm = np.hstack([np.arange(ncols, N, 2*(ncols+1)),
                            np.arange(2*ncols+1, N, 2*(ncols+1))])
        _prop = np.repeat(_prop, 2, axis=0)
        _prop = np.delete(_prop, rm, axis=0)
    else:
        ncols = ncols_odd
    _prop = _prop.reshape(nrows, ncols)

    # plotting maps
    if tiling == 'hex':
        color = (255*color[sel]).astype(int)
        x_hex = np.ndarray((len(x[sel]), 6))
        y_hex = np.ndarray((len(y[sel]), 6))

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
            hexagon = list(zip(*[x_hex[i], y_hex[i]]))
            draw.polygon(hexagon, fill=tuple(color[i]))

    elif tiling == 'rect':
        if grid.lower() == 'hexgrid':
            # double pixels
            sel = np.repeat(sel, 2)
            color = np.repeat(color, 2, axis=0)
            # remove extra pixels
            sel = np.delete(sel, rm, axis=0)
            color = np.delete(color, rm, axis=0)
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

        color = color.reshape(nrows, ncols, -1)

        img_pil = toimage(color[imin:imax, jmin:jmax, :])
        img_pil = img_pil.resize(size=(w, h))

    else:
        plt.close(fig)
        raise Exception('Unknown "{}" tiling'.format(tiling))

    ax.format_coord = _CoordsFormatter((x.min(), x.max(), y.max(), y.min()), _prop)
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


def plot_IPF(M, nrows, ncols_odd, ncols_even, x, y, dx=None, dy=None,
             d=[0, 0, 1], ax=None, sel=None, gray=None, grid='HexGrid',
             tiling=None, w=2048, scalebar=True, verbose=True, **kwargs):
    """
    Plots inverse pole figure map

    Parameters
    ----------
    M : numpy ndarray shape(N,3,3)
        Transformation matrix from the sample coordinate frame to
        the crystal coordinate frame.
    nrows : int
        Number of rows
    ncols_odd : int
        Number of columns in the odd rows
    ncols_even : int
        Number of columns in the even rows
    x : numpy ndarray shape(N)
        x pixel coordinates
    y : numpy ndarray shape(N)
        y pixel coordinates
    dx : float (optional)
        Grid spacing along x coordinates. If None is provided, guesses
        it from x.
        Default: None
    dy : float (optional)
        Grid spacing along y coordinates. If None is provided, guesses
        it from y.
        Default: None
    d : list or array shape(3)
        Reference direction in the sample coordinate frame.
        Default: [0, 0, 1] (i.e., normal direction)
    ax : AxesSubplot object (optional)
        The pole figure will be plotted in the provided object 'ax'
        Default: None
    sel : bool numpy 1D array (optional)
        Boolean array indicating which data points should be plotted
        Default: None
    gray : numpy ndarray (optional)
        Grayscale mask plotted over IPF.
        For example, one may want to overlay the IPF map with the image
        quality data.
        Default: None
    grid : str (optional)
        Grid type
        Default: 'HexGrid'
    tiling : str (optional)
        Valid options are 'rect' or 'hex'
        If no option is provided, uses as default 'rect' if 
        N > __THRESHOLD_TILING__, else 'hex'. By default, the value of
        __THRESHOLD_TILING__ is 10000, but it can be set to any value
        by calling pyebsd.set_threshold_tiling(..)
        Default: None
    w : int (optional)
        Width in pixel
        Default: 2048
    scalebar : bool (optional)
        If True, displays scalebar over IPF map
        Default: True
    verbose : bool (optional)
        If True, prints computation time
        Default: True

    **kwargs :
        kwargs parameters are passed to function ax.imshow:
        ax.imshow(img, ..., **kwargs)

    Returns
    -------
    ebsdmap : EBSDMap object
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
    if dy is None:
        dy = (np.max(y) - np.min(y))/(nrows - 1)

    if tiling == 'hex':
        edge_length = dx/3.**.5
        ymin -= edge_length/2.
        ymax += edge_length/2.
    else:
        ymin -= dy/2.
        ymax += dy/2.

    if grid.lower() == 'sqrgrid':
        xmin -= dx/2.
        xmax += dx/2.

    # getting kwargs parameters
    scalebar_location = kwargs.pop('scalebar_location', 'lower left')

    # min and max values of the gray mask
    graymin = kwargs.pop('graymin', None)
    graymax = kwargs.pop('graymax', None)

    # call IPF to get crystal directions parallel to d and
    # convert to color code (RGB)
    d_IPF = IPF(M, d)
    d_IPF = np.abs(d_IPF)
    d_IPF = np.sort(d_IPF, axis=1)
    d_IPF = d_IPF[:, [1, 0, 2]]
    color = get_color_IPF(d_IPF, issorted=True)
    # filling invalid/non-selected data points
    d_IPF[not_sel] = [np.nan, np.nan, np.nan]
    color[not_sel] = [0, 0, 0]  # RGB

    # applying gray mask
    if isinstance(gray, np.ndarray):
        if N != gray.shape[0]:
            raise Exception('N and len(gray) differ')
        else:
            if graymin is None:
                graymin = gray.min()
            if graymax is None:
                graymax = gray.max()
            gray = (gray.reshape(-1, 1) - graymin)/(graymax - graymin)
            gray[gray < 0.] = 0.
            gray[gray > 1.] = 1.
            color[sel] = color[sel]*gray[sel]

    # getting AxesSubplot object
    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.cla()
        fig = ax.get_figure()

    if grid.lower() == 'hexgrid':
        N, ncols = 2*N, 2*min(ncols_odd, ncols_even)  # N pixels and ncols for rect grid plotting
        if ncols_odd > ncols_even:
            rm = np.hstack([np.arange(0, N, 2*(ncols+1)),
                            np.arange(ncols+1, N, 2*(ncols+1))])
        else:
            rm = np.hstack([np.arange(ncols, N, 2*(ncols+1)),
                            np.arange(2*ncols+1, N, 2*(ncols+1))])
        d_IPF = np.repeat(d_IPF, 2, axis=0)
        d_IPF = np.delete(d_IPF, rm, axis=0)
    else:
        ncols = ncols_odd
    d_IPF = d_IPF.reshape(nrows, ncols, -1)

    # plotting maps
    if tiling == 'hex':
        color = color[sel]
        x_hex = np.ndarray((len(x[sel]), 6))
        y_hex = np.ndarray((len(y[sel]), 6))

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
            hexagon = list(zip(*[x_hex[i], y_hex[i]]))
            draw.polygon(hexagon, fill=tuple(color[i]))

    elif tiling == 'rect':
        if grid.lower() == 'hexgrid':
            # double pixels
            sel = np.repeat(sel, 2)
            color = np.repeat(color, 2, axis=0)
            # remove extra pixels
            sel = np.delete(sel, rm, axis=0)
            color = np.delete(color, rm, axis=0)
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

        color = color.reshape(nrows, ncols, -1)

        img_pil = toimage(color[imin:imax, jmin:jmax, :])
        img_pil = img_pil.resize(size=(w, h))

    else:
        plt.close(fig)
        raise Exception('Unknown "{}" tiling'.format(tiling))

    ax.format_coord = _CoordsFormatter((x.min(), x.max(), y.max(), y.min()), d_IPF.round(6), 'd')
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
