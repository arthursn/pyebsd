import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

from .orientation import euler_rotation
from .plotting import plot_property, plot_IPF, plot_PF

ssfonts = rcParams['font.sans-serif']


def _item2top(l, item):
    try:
        oldindex = l.index[item]
        l.insert(0, l.pop(oldindex))
    except:
        l.insert(0, item)
    return l


ssfonts = _item2top(ssfonts, 'Helvetica')
ssfonts = _item2top(ssfonts, 'Arial')

rcParams['font.sans-serif'] = ssfonts
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.0


class Scandata(object):
    __first_neighbors_hexgrid = np.array([
        [-2, 0],
        [-1, -1],
        [1, -1],
        [2, 0],
        [1, 1],
        [-1, 1]
    ], dtype=int)

    neighbors_hexgrid = [
        # 1st neighbors
        __first_neighbors_hexgrid,
        # 2nd neighbors
        np.array([
            [-3, -1],
            [0, -2],
            [3, -1],
            [3, 1],
            [0, 2],
            [-3, 1]
        ], dtype=int),
        # 3rd neighbors
        2*__first_neighbors_hexgrid,
        # 4th neighbors
        np.array([
            [-5, -1],
            [-4, -2],
            [-1, -3],
            [1, -3],
            [4, -2],
            [5, -1],
            [5, 1],
            [4, 2],
            [1, 3],
            [-1, 3],
            [-4, 2],
            [-5, 1],
        ], dtype=int),
        # 5th neighbors
        3*__first_neighbors_hexgrid
    ]

    def __init__(self, data, grid, dx, dy,
                 ncols_odd, ncols_even,
                 nrows, header=''):
        self.data = data  # pandas DataFrame
        self.grid = grid  # string (e.g., hexgrid)
        self.dx = dx  # float
        self.dy = dy  # float
        self.ncols_odd = ncols_odd  # int
        self.ncols_even = ncols_even  # int
        self.nrows = nrows  # int
        self.header = header  # string

        # total number of columns
        self.ncols = self.ncols_odd + self.ncols_even
        self.N = len(self.data)

        # .values: pandas Series to numpy array
        self.ind = self.data.index.values
        self.phi1 = self.data.phi1.values
        self.Phi = self.data.Phi.values
        self.phi2 = self.data.phi2.values
        self.x = self.data.x.values
        self.y = self.data.y.values
        self.IQ = self.data.IQ.values
        self.CI = self.data.CI.values
        self.ph = self.data.ph.values

        self._i = None  # row number
        self._j = None  # col number

        # R describes the rotation from the crystal base to the mechanical
        # coordinates of the EBSD system.
        self.R = euler_rotation(self.phi1, self.Phi, self.phi2)
        self._M = None

        self.figs_maps = []
        self.axes_maps = []

    @property
    def M(self):
        """
        M describes the rotation from the mechanical coordinates to the
        crystal base
        """
        if self._M is None:
            self._M = self.R.transpose([0, 2, 1])
        return self._M

    @property
    def i(self):
        """
        row number (0 -> nrows - 1)
        """
        if self._i is None:
            self._i = 2*(self.ind//self.ncols)
            shift = np.tile([0]*self.ncols_odd + [1] *
                            self.ncols_even, self.nrows)
            self._i += shift[:self.N]
        return self._i

    @property
    def j(self):
        """
        col number (0 -> ncols - 1)
        """
        if self._j is None:
            rem = self.ind % self.ncols  # remainder
            self._j = rem//self.ncols_odd + 2*(rem % self.ncols_odd)
        return self._j

    def ij2ind(self, i, j):
        """
        i, j grid positions to pixel index (self.ind)
        """
        # 1 - self.N*(j/self.ncols) turns negative every i, j pair where j > ncols
        return (1 - self.N*(j//self.ncols)) * \
            ((i//2)*self.ncols + (j % 2)*self.ncols_odd + (j//2))

    def get_neighbors(self, distance=1, perimeteronly=True):
        """
        Returns list of indices of the neighboring pixels for each pixel
        """

        if perimeteronly:
            j_shift, i_shift = self.neighbors_hexgrid[distance-1].T
        else:
            j_shift, i_shift = np.vstack(self.neighbors_hexgrid[:distance]).T

        j0, i0 = self.j, self.i
        # x
        j_neighbors = np.vstack(
            [j0 + shift for shift in j_shift]).T.astype(int)
        # y
        i_neighbors = np.vstack(
            [i0 + shift for shift in i_shift]).T.astype(int)

        neighbors_ind = self.ij2ind(i_neighbors, j_neighbors)
        neighbors_ind[(neighbors_ind < 0) | (neighbors_ind >= self.N)] = -1
        return neighbors_ind.astype(int)

    def plot_IPF(self, d='ND', ax=None, sel=None, gray=None, tiling='rect',
                 w=2048, scalebar=True, verbose=True, **kwargs):
        """
        Plot inverse pole figure

        Parameters
        ----------
        d : list or array shape(3) (optional)
            Mechanical direction parallel to the desired crystallographic direction A 
            string ['ND', ...] can provided instead. 'd' values will be assigned
            according to:
            'ND' : [0, 0, 1]
            Default: 'ND'
        ax : AxesSubplot instance (optional)
            The pole figure will be plotted in the provided instance 'ax'
        sel : boolean numpy 1D array
            Array with boolean [True, False] values indicating which data points 
            should be plotted
            Default: None
        gray : numpy ndarray (optional)
            Grayscale mask plotted over IPF. 
            For example, one may want to overlay the IPF map with the image 
            quality data.
            Default: None
        tiling : str (optional)
            Valid options are 'rect' or 'hex'
        w : int (optional)
            Width in pixel
            Default: 2048
        scalebar : booelan (optional)
            If True, displays scalebar over IPF map
            Default: True
        verbose : boolean (optional)
            If True, prints computation time
            Default: True

        **kwargs:
            Variables are passed to function ax.imshow:
            ax.imshow(img, ..., **kwargs)

        Returns
        -------
        ebsdmap : EBSDMap object

        """
        ebsdmap = plot_IPF(self.R, self.nrows, self.ncols_even, self.ncols_odd,
                           self.x, self.y, self.dx, d, ax, sel, gray, tiling, w,
                           scalebar, verbose, **kwargs)
        self.figs_maps.append(ebsdmap.ax.get_figure())
        self.axes_maps.append(ebsdmap.ax)
        return ebsdmap

    def plot_property(self, prop, ax=None, colordict=None, colorfill=[0, 0, 0, 1],
                      sel=None, gray=None, tiling='rect', w=2048,
                      scalebar=True, verbose=True, **kwargs):
        ebsdmap = plot_property(prop, self.nrows, self.ncols_even, self.ncols_odd,
                                self.x, self.y, self.dx, ax, colordict, colorfill,
                                sel, gray, tiling, w, scalebar, verbose, **kwargs)
        self.figs_maps.append(ebsdmap.ax.get_figure())
        self.axes_maps.append(ebsdmap.ax)
        return ebsdmap

    def plot_phase(self, ax=None, colordict={'1': [1, 0, 0, 1], '2': [0, 1, 0, 1]},
                   colorfill=[0, 0, 0, 1], sel=None, gray=None, tiling='rect',
                   w=2048, scalebar=True, verbose=True, **kwargs):
        ebsdmap = self.plot_property(self.ph, ax, colordict, colorfill, sel, gray,
                                     tiling, w, scalebar, verbose, **kwargs)
        self.figs_maps.append(ebsdmap.ax.get_figure())
        self.axes_maps.append(ebsdmap.ax)
        return ebsdmap

    def plot_PF(self, proj=[1, 0, 0], ax=None, sel=None, parent_or=None,
                contour=False, verbose=True, **kwargs):
        """
        Plot pole figure

        Parameters
        ----------
        proj : list or numpy array(3) (optional)
            Family of direction projected in the pole figure.
            Default: [1,0,0]
        ax : AxesSubplot instance (optional)
            The pole figure will be plotted in the provided instance 'ax'
        sel : boolean numpy ndarray
            Array with boolean [True, False] values indicating which data points 
            should be plotted
            Default: None
        parent_or : numpy ndarray shape(3, 3)
            Orientation matrix of the parent phase. The pole figure is rotated 
            until the axes coincides with the orientation 'parent_or'
            Default: None
        contour : boolean
            contour=True plots the pole figure using contour plot
            Default: False
        verbose : boolean
            If True, prints computation time
            Default: True

        **kwargs:
            lw_frame : float
                line width of PF frame
                Default: 0.5
            fill : [True, False]
                True: filled contour plot 'plt.contourf'; False: contour plot 
                'plt.contour'
                Default: True
            bins : int or tuple or array (int,int)
                Binning used in the calculation of the points density histogram (prior
                to contour plot)
                Default: (256, 256)
            fn : ['sqrt', 'log', 'None'] or function(x)
                function that modifies the points density.
                Default: 'sqrt'
            nlevels : int
                number of levels in the contour plot
                Default: 10

        The kwargs properties not listed here are automatically passed to the plotting
        functions:
        if not contour:
            plt.plot(..., **kwargs)
        if contour and fill:
            plt.contour(..., **kwargs)
        if contour and not fill:
            plt.contourf(..., **kwargs)

        Returns
        -------
        ax : matplotlib.pyplot.axes.Axes

        """
        return plot_PF(self.R, None, proj, ax, sel, parent_or, contour, verbose, **kwargs)

    def savefig(self, fname, **kwargs):
        kwargs.update({'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.0})
        plt.savefig(fname, **kwargs)


def _get_rectangle_surr_sel(scan, sel):
    """
    Select rectangle surrounding the selected data.
    Some manipulations are necessary to ensure that 
    ncols_odd = ncols_even + 1
    """
    # x
    ind_xmin = scan.data.x[sel].idxmin()
    ind_xmax = scan.data.x[sel].idxmax()
    # y
    ind_ymin = scan.data.y[sel].idxmin()
    ind_ymax = scan.data.y[sel].idxmax()

    # j
    jmin = scan.j[ind_xmin]
    jmax = scan.j[ind_xmax]
    # i
    imin = scan.i[ind_ymin]
    imax = scan.i[ind_ymax]

    if (jmin + imin) % 2 == 1:  # if jmin + imin is odd
        if jmin > 0:
            jmin -= 1  # add column to the left (jmin + imin has to be even)
        else:
            imin -= 1  # add row to the top

    if (jmax + imin) % 2 == 1:
        if jmax < scan.ncols - 1:
            jmax += 1  # add column to the right (jmax + imin has to be even)
        else:
            imin -= 1  # add row to the top
            jmin -= 1  # add [another] columns to the left

    xmin = scan.dx*(2.*jmin - 1.)/4.  # (jmin*dx/2 - dx/4)
    xmax = scan.dx*(2.*jmax + 1.)/4.  # (jmax*dx/2 + dx/4)
    ymin = scan.dy*(2.*imin - 1.)/2.  # (imin*dy - dy/2)
    ymax = scan.dy*(2.*imax + 1.)/2.  # (imax*dy + dy/2)

    ncols_even = (jmax - jmin)//2
    ncols_odd = ncols_even + 1
    nrows = imax - imin + 1

    # select rectangle surrounding the selected data
    rect = (scan.x >= xmin) & (scan.x <= xmax) & \
        (scan.y >= ymin) & (scan.y <= ymax)

    # total number of points
    N = ncols_even*(nrows//2) + ncols_odd*(nrows - nrows//2)
    if N != np.count_nonzero(rect):
        raise Exception(('Something went wrong: expected number '
                         'of points ({}) differs from what '
                         'we got ({})').format(N, np.count_nonzero(rect)))

    return ncols_odd, ncols_even, nrows, rect


def selection_to_scandata(scan, sel):
    """
    Convert selection to new Scandata object

    Arguments
    ---------
    scan : Scandata object
        Original Scandata object
    sel : numpy array
        array of booleans corresponding to the selection

    Returns
    -------
    newscan : Scandata object

    """

    # copy of scan.data numpy array to be exported
    newdata = scan.data.copy()  # raw data

    if sel is not None:
        # Regions not belonging to selection have values set to default
        newdata.loc[~sel, 'phi1'] = 4.
        newdata.loc[~sel, 'Phi'] = 4.
        newdata.loc[~sel, 'phi2'] = 4.
        newdata.loc[~sel, 'IQ'] = -1
        newdata.loc[~sel, 'CI'] = -2
        newdata.loc[~sel, 'ph'] = -1
        newdata.loc[~sel, 'intensity'] = -1
        newdata.loc[~sel, 'fit'] = 0

        # select rectangle surrounding the selected data
        ncols_odd, ncols_even, nrows, rect = _get_rectangle_surr_sel(scan, sel)

        # data to be exported is a rectangle
        newdata = newdata[rect]

        # offset x and y so (xmin, ymin) becomes the origin (0, 0)
        newdata.x -= newdata.x.min()
        newdata.y -= newdata.y.min()

    newscan = Scandata(newdata, scan.grid, scan.dx, scan.dy,
                       ncols_odd, ncols_even, nrows, scan.header)

    return newscan
