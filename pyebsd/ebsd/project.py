# -*- coding: utf-8 -*-

import numpy as np
from itertools import cycle
from matplotlib import rcParams
import matplotlib.pyplot as plt

from .orientation import euler_angles_to_rotation_matrix, misorientation, kernel_average_misorientation
from .plotting import plot_property, plot_IPF, plot_PF

__all__ = ['ScanData', 'selection_to_scandata']


def _item2top(l, item):
    try:
        oldindex = l.index[item]
        l.insert(0, l.pop(oldindex))
    except:
        l.insert(0, item)
    return l


ssfonts = rcParams['font.sans-serif']
ssfonts = _item2top(ssfonts, 'Helvetica')
ssfonts = _item2top(ssfonts, 'Arial')

rcParams['font.sans-serif'] = ssfonts
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.0


class ScanData(object):
    __supported_grids = ['hexgrid', 'sqrgrid']

    __cos60 = .5  # cos(60deg)
    __sin60 = .5*3.**.5  # sin(60deg)

    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']

    neighbors_hexgrid_fixed = [
        # 1st neighbors
        [[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1]],
        # 2nd neighbors
        [[3, 1], [0, 2], [-3, 1], [-3, -1], [0, -2], [3, -1]],
        # 3rd neighbors and so on...
        [[4, 0], [2, 2], [-2, 2], [-4, 0], [-2, -2], [2, -2]],
        [[5, 1], [4, 2], [1, 3], [-1, 3], [-4, 2], [-5, 1],
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

    def __init__(self, data, grid, dx, dy, ncols_odd, ncols_even, nrows, header=''):
        self.data = data  # pandas DataFrame
        self.grid = grid  # string (e.g., hexgrid)
        if self.grid.lower() not in self.__supported_grids:
            raise Exception('Unknown grid type "{}"'.format(self.grid))
        self.dx = dx  # float
        self.dy = dy  # float
        self.ncols_odd = ncols_odd  # int
        self.ncols_even = ncols_even  # int
        self.nrows = nrows  # int
        self.header = header  # string

        # total number of columns
        if self.grid.lower() == 'hexgrid':
            diff = abs(self.ncols_odd - self.ncols_even)
            if diff != 1:
                raise Exception('|ncols_odd - ncols_even| (|{} - {}| = {}) must be equal to 1'.format(
                    self.ncols_odd, self.ncols_even, diff))
            self.ncols = self.ncols_odd + self.ncols_even
        else:
            if self.ncols_odd != self.ncols_even:
                raise Exception('NCOLS_ODD ({}) and NCOLS_EVEN ({}) should be equal for {}'.format(
                    self.ncols_odd, self.ncols_even, self.grid))
            self.ncols = self.ncols_odd
        self.N = len(self.data)

        # .values: pandas Series to numpy array
        self.index = self.data.index.values
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

        # keeps history of Figure, AxesSubplot and EBSDMap objects in these
        # lists. self.clear_history() can be used to clear the history
        self.figs = []
        self.axes = []
        self.ebsdmaps = []

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
            if self.grid.lower() == 'hexgrid':
                self._i = 2*(self.index//self.ncols)
                shift = np.tile([0]*self.ncols_odd + [1]*self.ncols_even, self.nrows)
                self._i += shift[:self.N]
            else:
                self._i = self.index // self.ncols
        return self._i

    @property
    def j(self):
        """
        col number (0 -> ncols - 1)
        """
        if self._j is None:
            rem = self.index % self.ncols  # remainder
            if self.grid.lower() == 'hexgrid':
                rem_div = rem//self.ncols_odd
                rem_rem = rem % self.ncols_odd
                # special case
                if self.ncols_odd < self.ncols_even:
                    rem_div[self.ncols-1::self.ncols] = 1
                    rem_div = 1 - rem_div
                    rem_rem[self.ncols-1::self.ncols] = self.ncols_even - 1
                self._j = rem_div + 2*rem_rem
            else:
                self._j = rem
        return self._j

    def ij_to_index(self, i, j):
        """
        i, j grid positions to pixel index (self.index)

        Arguments
        ---------
        i : int
            Column number (y coordinate) according to grid description below
        j : int
            Row number (x coordinate) according to grid description below

        Returns
        -------
        index : int
            Pixel index

        Grid description for HexGrid:
        -----------------------------
        o : ncols_odd
        c : ncols_odd + ncols_even
        r : nrows
        n : total number of pixels

        ===================================
                     index
         0     1     2       o-2   o-1
         *     *     *  ...   *     *
            o    o+1            c-1
            *     *     ...      *
         c    c+1   c+2     c+o-2 c+o-1
         *     *     *  ...   *     *
                         .
                         .
                         .      n-1
            *     *     ...      *

        ===================================
                      j, i
         0  1  2  3  4   j         m-1
         *     *     *  ...   *     *   0

            *     *     ...      *      1

         *     *     *  ...   *     *   2
                         .
                         .              i
                         .
            *     *     ...      *     r-1

        Grid description for SqrGrid
        ----------------------------
        c : ncols_odd = ncols_even
        r : nrows
        n : total number of pixels

        ===================================
                     index
         0     1     2       c-2   c-1
         *     *     *  ...   *     *
         c    c+1   c+2     2c-2  2c-1
         *     *     *  ...   *     *
                         .
                         .
                         .   n-2   n-1
         *     *     *  ...   *     *

        ===================================
                      j, i
         0     1     2   j   n-2   n-1
         *     *     *  ...   *     *   0

         *     *     *        *     *   1
                         .
                         .              i
                         .
         *     *     *  ...   *     *  r-1

        """
        if self.grid.lower() == 'hexgrid':
            index = (i//2)*self.ncols + (j//2)
            # this is actually the normal situation
            if self.ncols_odd > self.ncols_even:
                index += (j % 2)*self.ncols_odd
            # this turns negative every i, j pair where j > ncols
            index *= (1 - self.N*(j//self.ncols))
        else:
            index = i*self.ncols + j
        return index

    def get_neighbors_oim(self, distance):
        """
        Returns list of relative indices of the neighboring pixels for
        a given distance in pixels
        """
        if self.grid.lower() == 'hexgrid':
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
        else:  # sqrgrid
            R90 = np.array([[0, -1],
                            [1,  0]], dtype=int)  # 90 degrees rotation matrix
            xy = np.vstack([np.arange(-distance, distance, dtype=int),
                            np.full(2*distance, -distance, dtype=int)])

            j_list, i_list = list(xy[0]), list(xy[1])

            for r in range(1, 4):
                xy = np.dot(R90, xy)
                j_list += list(xy[0])
                i_list += list(xy[1])

        return j_list, i_list

    def get_neighbors_fixed(self, distance):
        """
        Returns list of relative indices of the neighboring pixels for
        a given distance in pixels
        """
        if self.grid.lower() == 'hexgrid':
            if distance > self.__n_neighbors_hexgrid_fixed:
                raise Exception('get_neighbors_fixed not supported for distance > {}'.format(
                    self.__n_neighbors_hexgrid_fixed))
            j_list, i_list = list(zip(*self.neighbors_hexgrid_fixed[distance-1]))
        else:
            raise Exception('get_neighbors_fixed not yet supported for grid type {}'.format(self.grid))
        return list(j_list), list(i_list)

    def get_neighbors(self, distance, perimeteronly=True, distance_convention='OIM'):
        """
        Returns list of indices of the neighboring pixels for every pixel
        for a given distance in pixels
        """
        if distance_convention.lower() == 'oim':
            _get_neighbors = self.get_neighbors_oim
        elif distance_convention.lower() == 'fixed':
            _get_neighbors = self.get_neighbors_fixed
        else:
            raise Exception('get_neighbors: unknown distance convention "{}"'.format(distance_convention))

        if perimeteronly:
            # only pixels in the perimeter
            j_shift, i_shift = _get_neighbors(distance)
        else:
            # including inner pixels
            j_shift, i_shift = [], []
            for d in range(1, distance+1):
                j_sh, i_sh = _get_neighbors(d)
                j_shift += j_sh
                i_shift += i_sh

        # j, i indices as np arrays
        j0, i0 = self.j, self.i
        # x
        j_neighbors = np.vstack([j0 + shift for shift in j_shift]).T.astype(int)
        # y
        i_neighbors = np.vstack([i0 + shift for shift in i_shift]).T.astype(int)
        outliers = (j_neighbors < 0) | (j_neighbors >= self.ncols) | (i_neighbors < 0) | (i_neighbors >= self.nrows)

        neighbors_ind = self.ij_to_index(i_neighbors, j_neighbors)
        neighbors_ind[outliers] = -1

        return neighbors_ind.astype(int)

    def get_distance_neighbors(self, distance, distance_convention='OIM'):
        if distance_convention.lower() == 'oim':
            j, i = self.get_neighbors_oim(distance)
        elif distance_convention.lower() == 'fixed':
            j, i = self.get_neighbors_fixed(distance)
        else:
            raise Exception('get_distance_neighbors: unknown distance convention "{}"'.format(distance_convention))

        if self.grid.lower() == 'hexgrid':
            d = .5*(np.array(j)**2 + 3.*np.array(i)**2)**.5
        else:  # sqrgrid
            d = (np.array(j)**2 + np.array(i)**2)**.5

        return d.mean()

    def get_KAM(self, distance=1, perimeteronly=True, maxmis=None,
                distance_convention='OIM', sel=None, **kwargs):
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
        return kernel_average_misorientation(self.M, neighbors, sel, maxmis, **kwargs)

    def plot_IPF(self, d=[0, 0, 1], ax=None, sel=None, gray=None, tiling=None,
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

        **kwargs:
            Variables are passed to function ax.imshow:
            ax.imshow(img, ..., **kwargs)

        Returns
        -------
        ebsdmap : EBSDMap object

        """
        ebsdmap = plot_IPF(self.M, self.nrows, self.ncols_even, self.ncols_odd, self.x, self.y,
                           self.dx, self.dy, d, ax, sel, gray, self.grid, tiling, w, scalebar,
                           verbose, **kwargs)
        self.ebsdmaps.append(ebsdmap)
        self.figs.append(ebsdmap.ax.get_figure())
        self.axes.append(ebsdmap.ax)
        return ebsdmap

    def plot_property(self, prop, ax=None, colordict=None, colorfill='black',
                      fillvalue=np.nan, sel=None, gray=None, tiling=None, w=2048,
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

        **kwargs:
            Variables are passed to function ax.imshow:
            ax.imshow(img, ..., **kwargs)

        Returns
        -------
        ebsdmap : EBSDMap object

        """
        ebsdmap = plot_property(prop, self.nrows, self.ncols_even, self.ncols_odd, self.x, self.y,
                                self.dx, self.dy, ax, colordict, colorfill, fillvalue, sel, gray,
                                self.grid, tiling, w, scalebar, colorbar, verbose, **kwargs)
        self.ebsdmaps.append(ebsdmap)
        self.figs.append(ebsdmap.ax.get_figure())
        self.axes.append(ebsdmap.ax)
        return ebsdmap

    def plot_phase(self, ax=None, colordict=None,
                   colorfill='black', fillvalue=-1, sel=None, gray=None,
                   tiling=None, w=2048, scalebar=True, verbose=True, **kwargs):
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
            Default: -1
        gray : numpy ndarray (optional)
            Grayscale mask plotted over IPF.
            For example, one may want to overlay the IPF map with the image
            quality data.
            Default: None
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

        **kwargs:
            Variables are passed to function ax.imshow:
            ax.imshow(img, ..., **kwargs)

        Returns
        -------
        ebsdmap : EBSDMap object

        """
        if colordict is None:
            ph_code = set(self.ph)
            ccycler = cycle(self.colors)
            colordict = {ph: next(ccycler) for ph in ph_code}
        ebsdmap = self.plot_property(self.ph, ax, colordict, colorfill, fillvalue, sel, gray,
                                     tiling, w, scalebar, False, verbose, **kwargs)
        self.ebsdmaps.append(ebsdmap)
        self.figs.append(ebsdmap.ax.get_figure())
        self.axes.append(ebsdmap.ax)
        return ebsdmap

    def plot_KAM(self, distance=1, perimeteronly=True, ax=None, maxmis=None,
                 distance_convention='OIM', colorfill='black', fillvalue=np.nan,
                 sel=None, gray=None, tiling=None, w=2048, scalebar=True, colorbar=True,
                 verbose=True, **kwargs):
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
        distance_convention : str (optional)
            Valid options are 'OIM' and 'fixed'
            Convention used for calculating distance to kernel
            Default: OIM
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

        **kwargs:
            Variables are passed to function ax.imshow:
            ax.imshow(img, ..., **kwargs)

        Returns
        -------
        ebsdmap : EBSDMap object

        """
        KAM = self.get_KAM(distance, perimeteronly, maxmis, distance_convention, sel)
        ebsdmap = self.plot_property(KAM, ax, None, colorfill, fillvalue, sel, gray,
                                     tiling, w, scalebar, colorbar, verbose, **kwargs)
        ebsdmap.cax.set_label(u'KAM (°)')
        self.ebsdmaps.append(ebsdmap)
        self.figs.append(ebsdmap.ax.get_figure())
        self.axes.append(ebsdmap.ax)
        return ebsdmap

    def plot_PF(self, proj=[1, 0, 0], ax=None, sel=None, rotation=None, contour=False,
                verbose=True, **kwargs):
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
        self.figs[-1].savefig(fname, **kwargs)

    def save_ang_file(self, fname, sel=None, **kwargs):
        """
        Export ScanData as ang file

        Arguments
        ---------
        fname : string
            File name
        sel : list of array of booleans
            selection
        """

        if sel is None:
            newscan = self
        else:
            newscan = selection_to_scandata(self, sel)

        header = newscan.header
        for i, line in enumerate(header):
            if '# NCOLS_ODD:' in line:
                header[i] = '# NCOLS_ODD: {:d}\n'.format(newscan.ncols_odd)
                continue
            if '# NCOLS_EVEN:' in line:
                header[i] = '# NCOLS_EVEN: {:d}\n'.format(newscan.ncols_even)
                continue
            if '# NROWS:' in line:
                header[i] = '# NROWS: {:d}\n'.format(newscan.nrows)
                continue

        try:
            file = open(fname, 'w')
            file.write(''.join(header))
            file.close()
            newscan.data.to_csv(fname, mode='a',
                                header=False, index=False, sep=' ',
                                float_format=kwargs.pop('float_format', '%.5f'))
        except:
            raise
        else:
            print('scandata successfully saved as "{}"'.format(fname))

    def clear_history(self):
        """
        Closes all figure windows and clear history of Figure, AxesSubplot
        and EBSDMap objects
        """
        for fig in self.figs:
            plt.close(fig)
        del self.ebsdmaps[:]
        del self.figs[:]
        del self.axes[:]


def _get_rectangle_surrounding_selection_hexgrid(scan, sel):
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
    if scan.grid.lower() == 'hexgrid':
        ncols_odd, ncols_even, nrows, rect = _get_rectangle_surrounding_selection_hexgrid(scan, sel)
    else:
        raise Exception('selection_to_scandata not yet supported for grid type {}'.format(scan.grid))

    # data to be exported is a rectangle
    newdata = newdata[rect]

    # offset x and y so (xmin, ymin) becomes the origin (0, 0)
    newdata.x -= newdata.x.min()
    newdata.y -= newdata.y.min()

    return ScanData(newdata, scan.grid, scan.dx, scan.dy, ncols_odd, ncols_even, nrows, scan.header)
