# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

from .orientation import euler_angles_to_rotation_matrix, misorientation
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


class ScanData(object):
    __cos60 = .5  # cos(60deg)
    __sin60 = .5*3.**.5  # sin(60deg)

    neighbors_hexgrid_fixed = [
        # 1st neighbors
        [[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1]],
        # 2nd neighbors
        [[3, 1], [0, 2], [-3, 1], [-3, -1], [0, -2], [3, -1]],
        # 3rd neighbors and so on...
        [[4, 0], [2, 2], [-2, 2], [-4, 0], [-2, -2], [2, -2]],
        [[5, -1], [4, 2], [1, 3], [-1, 3], [-4, 2], [-5, 1],
         [-5, -1], [-4, -2], [-1, -3], [1, -3], [4, -2], [5, -1]],
        [[6, 0], [3, 3], [-3, 3], [-6, 0], [-3, -3], [3, -3]],
        [[6, 2], [0, 4], [-6, 2], [-6, -2], [0, -4], [6, -2]],
        [[7, 1], [5, 3], [2, 4], [-2, 4], [-5, 3], [-7, 1],
         [-7, -1], [-5, -3], [-2, -4], [2, -4], [5, -3], [7, -1]],
        [[8, 0], [4, 4], [-4, 4], [-8, 0], [-4, -4], [4, -4]],
        [[8, 2], [7, 3], [1, 5], [-1, 5], [-7, 3], [-8, 2],
         [-8, -2], [-7, -3], [-1, -5], [1, -5], [7, -3], [8, -2]],
        [[9, 1], [6, 4], [3, 5], [-3, 5], [-6, 4], [-9, 1],
         [-9, -1], [-6, -4], [-3, -5], [3, -5], [6, -4], [9, -1]],
        [[10, 0], [5, 5], [-5, 5], [-10, 0], [-5, -5], [5, -5]],
        [[9, 3], [0, 6], [-9, 3], [-9, -3], [0, -6], [9, -3]],
        [[10, 2], [8, 4], [2, 6], [-2, 6], [-8, 4], [-10, 2],
         [-10, -2], [-8, -4], [-2, -6], [2, -6], [8, -4], [10, -2]],
        [[11, 1], [7, 5], [4, 6], [-4, 6], [-7, 5], [-11, 1],
         [-11, -1], [-7, -5], [-4, -6], [4, -6], [7, -5], [11, -1]],
        # 15th neighbors
        [[12, 0], [6, 6], [-6, 6], [-12, 0], [-6, -6], [6, -6]]
    ]

    __n_neighbors_hexgrid_fixed = len(neighbors_hexgrid_fixed)

    def __init__(self, data, grid, dx, dy, ncols_odd, ncols_even,
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

        self._M = None
        self._R = None

        self.figs_maps = []
        self.axes_maps = []

    @property
    def M(self):
        """
        M describes the rotation from the sample coordinate frame to the
        crystal coordinate frame
        """
        if self._M is None:
            self._M = self.R.transpose([0, 2, 1])
        return self._M

    @property
    def R(self):
        """
        R describes the rotation from the crystal coordinate frame to the
        sample coordinate frame of the EBSD system.
        """
        if self._R is None:
            self._R = euler_angles_to_rotation_matrix(
                self.phi1, self.Phi, self.phi2)
        return self._R

    @property
    def i(self):
        """
        row number (0 -> nrows - 1)
        """
        if self._i is None:
            self._i = 2*(self.ind//self.ncols)
            shift = np.tile([0]*self.ncols_odd + [1]*self.ncols_even, self.nrows)
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

        n : ncols_odd
        m : ncols_even
        N : ncols_odd + ncols_even

        ----------------------------------------
                  indices ind
         0     1     2       n-2   n-1
         *     *     *  ...   *     *
            n    n+1            N-1
            *     *              *     
         N    N+1   N+2
         *     *     *  ...   *     *
                         .
                         .
                         .

            *     *     ...      *

         *     *     *  ...   *     *

        ----------------------------------------
                   columns j
         0  1  2  3  4             N-1
         *     *     *  ...   *     *   (row 0)

            *     *              *      (row 1)

         *     *     *  ...   *     *   (row 2)
                         .
                         .
                         .

            *     *     ...      *      (row i)

         *     *     *  ...   *     *   (row i+1)

        """
        # 1 - self.N*(j/self.ncols) turns negative every i, j
        # pair where j > ncols
        return (1 - self.N*(j//self.ncols)) * \
            ((i//2)*self.ncols + (j % 2)*self.ncols_odd + (j//2))

    def get_neighbors_hexgrid_oim(self, distance):
        """
        Returns list of relative indices of the neighboring pixels in a
        hexgrid for a given distance in pixels
        """
        R60 = np.array([[self.__cos60, -self.__sin60],
                        [self.__sin60,  self.__cos60]])  # 60 degrees rotation matrix

        j_list = np.arange(-distance, distance, 2)
        i_list = np.full(j_list.shape, -distance)

        xy = np.vstack([j_list*self.__cos60, i_list*self.__sin60])

        j_list, i_list = list(j_list), list(i_list)

        for r in range(1, 6):
            xy = np.dot(R60, xy)  # 60 degrees rotation
            j_list += list((xy[0]/self.__cos60).round(0).astype(int))
            i_list += list((xy[1]/self.__sin60).round(0).astype(int))

        return j_list, i_list

    def get_neighbors_hexgrid_fixed(self, distance):
        """
        Returns list of relative indices of the neighboring pixels in a
        hexgrid for a given distance in pixels
        """
        if distance > self.__n_neighbors_hexgrid_fixed:
            raise Exception('Not supported for distance > {}'.format(self.__n_neighbors_hexgrid_fixed))

        j_list, i_list = list(zip(*self.neighbors_hexgrid_fixed[distance-1]))
        return list(j_list), list(i_list)

    def get_neighbors(self, distance, perimeteronly=True, distance_convention='OIM'):
        """
        Returns list of indices of the neighboring pixels for every pixel
        for a given distance in pixels
        """
        if distance_convention.lower() == 'oim':
            get_neighbors_hexgrid = self.get_neighbors_hexgrid_oim
        elif distance_convention.lower() == 'fixed':
            get_neighbors_hexgrid = self.get_neighbors_hexgrid_fixed
        else:
            raise Exception('Invalid distance convention "{}"'.format(distance_convention))

        if perimeteronly:
            # only pixels in the perimeter
            j_shift, i_shift = get_neighbors_hexgrid(distance)
        else:
            # including inner pixels
            j_shift, i_shift = [], []
            for d in range(1, distance+1):
                j_sh, i_sh = get_neighbors_hexgrid(d)
                j_shift += j_sh
                i_shift += i_sh

        # j, i indices as np arrays
        j0, i0 = self.j, self.i
        # x
        j_neighbors = np.vstack([j0 + shift for shift in j_shift]).T.astype(int)
        # y
        i_neighbors = np.vstack([i0 + shift for shift in i_shift]).T.astype(int)

        neighbors_ind = self.ij2ind(i_neighbors, j_neighbors)
        neighbors_ind[(neighbors_ind < 0) | (neighbors_ind >= self.N)] = -1
        return neighbors_ind.astype(int)

    def get_distance_neighbors(self, distance, distance_convention='OIM'):
        if distance_convention.lower() == 'oim':
            j, i = self.get_neighbors_hexgrid_oim(distance)
        elif distance_convention.lower() == 'fixed':
            j, i = self.get_neighbors_hexgrid_fixed(distance)
        else:
            raise Exception('Invalid distance convention "{}"'.format(distance_convention))

        d = .5*(np.array(j)**2. + 3.*np.array(i)**2.)**.5
        return d.mean()

    def get_KAM(self, distance=1, perimeteronly=True, maxmis=None,
                distance_convention='OIM', sel=None):
        """
        Returns Kernel average misorientation map

        Parameters
        ----------
        distance : int (optional)
            Distance (in neighbor indexes) to the kernel
            Default: 1
        perimeteronly : bool (optional)
            If True, KAM is calculated using only pixels in the perimeter,
            else uses inner pixels as well
            Default: True
        maxmis : float (optional)
            Maximum misorientation angle (in degrees) accounted in the 
            calculation of KAM
            Default: None
        sel : bool numpy 1D array (optional)
            Boolean array indicating which data points should be plotted
            Default: None

        Returns
        -------
        KAM : np.ndarray(N) with KAM values in degrees

        """
        neighbors = self.get_neighbors(distance, perimeteronly, distance_convention)
        misang = misorientation(self.M, neighbors, sel)

        if maxmis is not None:
            misang[misang > maxmis] = 0
            nneighbors = np.count_nonzero(misang <= maxmis, axis=1)
            nneighbors[nneighbors == 0] = 1  # to prevent division by 0
        else:
            nneighbors = neighbors.shape[1]

        kam = np.sum(misang, axis=1)/nneighbors

        if maxmis is not None:
            kam[nneighbors == 0] = np.nan

        return kam

    def plot_IPF(self, d=[0, 0, 1], ax=None, sel=None, gray=None, tiling='rect',
                 w=2048, scalebar=True, verbose=True, **kwargs):
        """
        Plots inverse pole figure map

        Parameters
        ----------
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
        tiling : str (optional)
            Valid options are 'rect' or 'hex'
        w : int (optional)
            Width in pixel
            Default: 2048
        scalebar : bool (optional)
            If True, displays scalebar over IPF map
            Default: True
        verbose : bool (optional)
            If True, prints computation time
            Default: True

        **kwargs:
            Variables are passed to function ax.imshow:
            ax.imshow(img, ..., **kwargs)

        Returns
        -------
        ebsdmap : EBSDMap object

        """
        ebsdmap = plot_IPF(self.M, self.nrows, self.ncols_even, self.ncols_odd,
                           self.x, self.y, self.dx, d, ax, sel, gray, tiling, w,
                           scalebar, verbose, **kwargs)
        self.figs_maps.append(ebsdmap.ax.get_figure())
        self.axes_maps.append(ebsdmap.ax)
        return ebsdmap

    def plot_property(self, prop, ax=None, colordict=None, colorfill=[0, 0, 0, 1],
                      sel=None, gray=None, tiling='rect', w=2048,
                      scalebar=True, colorbar=True, verbose=True, **kwargs):
        """
        Plots any EBSD property

        Parameters
        ----------
        prop : array shape(N)
            Property to be plotted provided as np.ndarray(N), where N is the 
            size of the data file
        ax : AxesSubplot object (optional)
            The pole figure will be plotted in the provided object 'ax'
            Default: None
        colordict : dict(str: list shape(4)) (optional)
            Dictionary that maps indexed phase to respective color provided
            as list shape(4) (RGBA from 0 to 1)
            Default: None
        colorfill : list shape(4) (optional)
            Color used to fill unindexed pixels
            Default: [0, 0, 0, 1] (black)
        sel : bool numpy 1D array (optional)
            Boolean array indicating which data points should be plotted
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
        scalebar : bool (optional)
            If True, displays scalebar over IPF map
            Default: True
        verbose : bool (optional)
            If True, prints computation time
            Default: True

        **kwargs:
            Variables are passed to function ax.imshow:
            ax.imshow(img, ..., **kwargs)

        Returns
        -------
        ebsdmap : EBSDMap object

        """
        ebsdmap = plot_property(prop, self.nrows, self.ncols_even, self.ncols_odd,
                                self.x, self.y, self.dx, ax, colordict, colorfill,
                                sel, gray, tiling, w, scalebar, colorbar, verbose, **kwargs)
        self.figs_maps.append(ebsdmap.ax.get_figure())
        self.axes_maps.append(ebsdmap.ax)
        return ebsdmap

    def plot_phase(self, ax=None, colordict={'1': [1, 0, 0, 1], '2': [0, 1, 0, 1]},
                   colorfill=[0, 0, 0, 1], sel=None, gray=None, tiling='rect',
                   w=2048, scalebar=True, verbose=True, **kwargs):
        """
        Plots phases map 

        Parameters
        ----------
        ax : AxesSubplot object (optional)
            The pole figure will be plotted in the provided object 'ax'
            Default: None
        sel : bool numpy 1D array (optional)
            Boolean array indicating which data points should be plotted
            Default: None
        colordict : dict(str: list shape(4)) (optional)
            Dictionary that maps indexed phase to respective color provided
            as list shape(4) (RGBA from 0 to 1)
            Default: {'1': [1, 0, 0, 1], '2': [0, 1, 0, 1]}
        colorfill : list shape(4) (optional)
            Color used to fill unindexed pixels
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
        scalebar : bool (optional)
            If True, displays scalebar over IPF map
            Default: True
        verbose : bool (optional)
            If True, prints computation time
            Default: True

        **kwargs:
            Variables are passed to function ax.imshow:
            ax.imshow(img, ..., **kwargs)

        Returns
        -------
        ebsdmap : EBSDMap object

        """
        ebsdmap = self.plot_property(self.ph, ax, colordict, colorfill, sel, gray,
                                     tiling, w, scalebar, False, verbose, **kwargs)
        self.figs_maps.append(ebsdmap.ax.get_figure())
        self.axes_maps.append(ebsdmap.ax)
        return ebsdmap

    def plot_KAM(self, distance=1, perimeteronly=True, ax=None, maxmis=None,
                 distance_convention='OIM', sel=None, gray=None, tiling='rect',
                 w=2048, scalebar=True, colorbar=True, verbose=True, **kwargs):
        """
        Plots kernel average misorientation map

        Parameters
        ----------
        distance : int (optional)
            Distance (in neighbor indexes) to the kernel
            Default: 1
        perimeteronly : bool (optional)
            If True, KAM is calculated using only pixels in the perimeter,
            else uses inner pixels as well
            Default: True
        ax : AxesSubplot object (optional)
            The pole figure will be plotted in the provided object 'ax'
            Default: None
        maxmis : float (optional)
            Maximum misorientation angle (in degrees) accounted in the 
            calculation of KAM
            Default: None
        sel : bool numpy 1D array (optional)
            Boolean array indicating which data points should be plotted
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
        scalebar : bool (optional)
            If True, displays scalebar over IPF map
            Default: True
        verbose : bool (optional)
            If True, prints computation time
            Default: True

        **kwargs:
            Variables are passed to function ax.imshow:
            ax.imshow(img, ..., **kwargs)

        Returns
        -------
        ebsdmap : EBSDMap object

        """
        KAM = self.get_KAM(distance, perimeteronly, maxmis, distance_convention, sel)
        ebsdmap = self.plot_property(KAM, ax, None, [0, 0, 0, 1], sel, gray,
                                     tiling, w, scalebar, colorbar, verbose, **kwargs)
        ebsdmap.cax.set_label(u'KAM (°)')
        self.figs_maps.append(ebsdmap.ax.get_figure())
        self.axes_maps.append(ebsdmap.ax)
        return ebsdmap

    def plot_PF(self, proj=[1, 0, 0], ax=None, sel=None, rotation=None,
                contour=False, verbose=True, **kwargs):
        """
        Plots pole figure

        Parameters
        ----------
        proj : list or numpy array(3) (optional)
            Family of direction projected in the pole figure.
            Default: [1,0,0]
        ax : AxesSubplot object (optional)
            The pole figure will be plotted in the provided object 'ax'
        sel : bool numpy ndarray
            Array with bool [True, False] values indicating which data 
            points should be plotted
            Default: None
        rotation : list or array shape(3,3)
            Rotation matrix that rotates the pole figure.
            The columns of the matrix correspond to the directions parallel to 
            the axes of the pole figure.
        contour : bool (optional)
            contour=True plots the pole figure using contour plot
            Default: False
        verbose : bool (optional)
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
                Binning used in the calculation of the points density histogram 
                (prior to contour plot)
                Default: (256, 256)
            fn : ['sqrt', 'log', 'None'] or function(x)
                function that modifies the points density.
                Default: 'sqrt'
            nlevels : int
                number of levels in the contour plot
                Default: 10

        The kwargs properties not listed here are automatically passed to the 
        plotting functions:
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
        return plot_PF(None, proj, ax, sel, rotation, contour, verbose, R=self.R, **kwargs)

    def savefig(self, fname, **kwargs):
        """
        Saves ebsd map plotted last
        """
        kwargs.update({'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.0})
        plt.savefig(fname, **kwargs)


def _get_rectangle_surrounding_selection(scan, sel):
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
    Convert selection to new ScanData object

    Arguments
    ---------
    scan : ScanData object
        Original ScanData object
    sel : numpy array
        array of booleans corresponding to the selection

    Returns
    -------
    newscan : ScanData object

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
        ncols_odd, ncols_even, nrows, rect = _get_rectangle_surrounding_selection(
            scan, sel)

        # data to be exported is a rectangle
        newdata = newdata[rect]

        # offset x and y so (xmin, ymin) becomes the origin (0, 0)
        newdata.x -= newdata.x.min()
        newdata.y -= newdata.y.min()

    newscan = ScanData(newdata, scan.grid, scan.dx, scan.dy,
                       ncols_odd, ncols_even, nrows, scan.header)

    return newscan
