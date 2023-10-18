# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib import rcParams
from matplotlib.axes import Axes
from dataclasses import dataclass

from .orientation import euler_angles_to_rotation_matrix, kernel_average_misorientation
from .plotting import GridIndexing, EBSDMap, plot_property, plot_IPF, plot_PF

__all__ = ["ScanData", "selection_to_scandata"]


rcParams["savefig.dpi"] = 300
rcParams["savefig.bbox"] = "tight"
rcParams["savefig.pad_inches"] = 0.0


class ScanData(GridIndexing):
    """
    EBSD scan data

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the EBSD data. It is compulsory for data to
        contain the following columns: phi1, Phi, phi2 (Euler angles), x,
        y (pixel coordinates), and ph (phase code). These columns are
        then parsed as data members of the ScanData object. If IQ and CI
        are also provided as columns, they are also parsed.
    grid : str
        Grid type. Possible options are 'HexGrid' and 'SqrGrid'
    dx : float
        Grid spacing along x coordinate
    dy : float
        Grid spacing along y coordinate
    ncols_odd : int
        Number of columns in the odd rows (first row is indexed as 1, odd)
    ncols_even : int
        Number of columns in the even rows
    nrows : int
        Number of rows
    header : str (optional)
        Header of the scan data file
        Default: ''
    """

    _2pi = 2 * np.pi
    _cos60 = 0.5  # cos(60deg)
    _sin60 = 0.5 * 3.0**0.5  # sin(60deg)

    colors = ["red", "green", "blue", "cyan", "magenta", "yellow"]

    neighbors_hexgrid_fixed = [
        # 1st neighbors
        [[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1]],
        # 2nd neighbors
        [[3, 1], [0, 2], [-3, 1], [-3, -1], [0, -2], [3, -1]],
        # 3rd neighbors and so on...
        [[4, 0], [2, 2], [-2, 2], [-4, 0], [-2, -2], [2, -2]],
        [
            [5, 1],
            [4, 2],
            [1, 3],
            [-1, 3],
            [-4, 2],
            [-5, 1],
            [-5, -1],
            [-4, -2],
            [-1, -3],
            [1, -3],
            [4, -2],
            [5, -1],
        ],
        [[6, 0], [3, 3], [-3, 3], [-6, 0], [-3, -3], [3, -3]],
        [[6, 2], [0, 4], [-6, 2], [-6, -2], [0, -4], [6, -2]],
        [
            [7, 1],
            [5, 3],
            [2, 4],
            [-2, 4],
            [-5, 3],
            [-7, 1],
            [-7, -1],
            [-5, -3],
            [-2, -4],
            [2, -4],
            [5, -3],
            [7, -1],
        ],
        [[8, 0], [4, 4], [-4, 4], [-8, 0], [-4, -4], [4, -4]],
        [
            [8, 2],
            [7, 3],
            [1, 5],
            [-1, 5],
            [-7, 3],
            [-8, 2],
            [-8, -2],
            [-7, -3],
            [-1, -5],
            [1, -5],
            [7, -3],
            [8, -2],
        ],
        [
            [9, 1],
            [6, 4],
            [3, 5],
            [-3, 5],
            [-6, 4],
            [-9, 1],
            [-9, -1],
            [-6, -4],
            [-3, -5],
            [3, -5],
            [6, -4],
            [9, -1],
        ],
        [[10, 0], [5, 5], [-5, 5], [-10, 0], [-5, -5], [5, -5]],
        [[9, 3], [0, 6], [-9, 3], [-9, -3], [0, -6], [9, -3]],
        [
            [10, 2],
            [8, 4],
            [2, 6],
            [-2, 6],
            [-8, 4],
            [-10, 2],
            [-10, -2],
            [-8, -4],
            [-2, -6],
            [2, -6],
            [8, -4],
            [10, -2],
        ],
        [
            [11, 1],
            [7, 5],
            [4, 6],
            [-4, 6],
            [-7, 5],
            [-11, 1],
            [-11, -1],
            [-7, -5],
            [-4, -6],
            [4, -6],
            [7, -5],
            [11, -1],
        ],
        # 15th neighbors
        [[12, 0], [6, 6], [-6, 6], [-12, 0], [-6, -6], [6, -6]],
    ]

    _n_neighbors_hexgrid_fixed = len(neighbors_hexgrid_fixed)

    @dataclass
    class FieldNameInfo:
        aliases: list
        copy: bool = False

    _fieldnames = {
        # Compulsory
        "x": FieldNameInfo(aliases=["X"], copy=True),
        "y": FieldNameInfo(aliases=["Y"], copy=True),
        "phi1": FieldNameInfo(aliases=["Euler1"], copy=False),
        "Phi": FieldNameInfo(aliases=["Euler2"], copy=False),
        "phi2": FieldNameInfo(aliases=["Euler3"], copy=False),
        "phase": FieldNameInfo(aliases=["ph", "Phase"], copy=False),
        # Optional
        "IQ": FieldNameInfo(aliases=["Band Contrast"], copy=False),
    }
    _compulsory_columns = ["x", "y", "phi1", "Phi", "phi2", "phase"]

    def __init__(self, data, grid, dx, dy, ncols_odd, ncols_even, nrows, header=""):
        # Initializes base class GridIndexing
        super(ScanData, self).__init__(grid, ncols_odd, ncols_even, nrows, dx, dy)

        self.data = data  # pandas DataFrame
        if len(data) != self.N:
            raise Exception(
                ("Number of pixels ({}) does not match expected value " "({})").format(
                    len(data), self.N
                )
            )

        # Rename columns in dataframe
        self.header = header  # string

        # Set compulsory columns in data (x, y, phi1 ...) as member variables
        # (.values: pandas Series to numpy array)
        for field_name, field_name_info in self._fieldnames.items():
            for cname in self.data.columns:
                if cname == field_name or cname in field_name_info.aliases:
                    if not field_name_info.copy:
                        setattr(self, field_name, self.data[cname].values)
                    else:
                        setattr(self, field_name, self.data[cname].values.copy())
                    # Set aliases as attributes
                    for alias in field_name_info.aliases:
                        # Just in case... This should never happen
                        if alias == field_name:
                            continue
                        setattr(self, alias, getattr(self, field_name))

        if (
            abs(self.phi1.max()) > self._2pi
            or abs(self.Phi.max()) > self._2pi
            or abs(self.phi2.max()) > self._2pi
        ):
            print(
                "Euler angles out of allowed range! Please check if they are really "
                "provided in radians."
            )

        # Makes sure min(x) == 0 and min(y) == 0
        self.x -= self.x.min()
        self.y -= self.y.min()

        # Optional columns
        for colname, coldata in self.data.items():
            if colname not in self._compulsory_columns and isinstance(colname, str):
                try:
                    # Check if colname is a valid variable name. Only works on python3,
                    # that's why the exception handling
                    if colname.isidentifier():
                        setattr(self, colname, coldata.values)
                except AttributeError:
                    setattr(self, colname, coldata.values)

        self._i = None  # row number
        self._j = None  # col number

        self._M = None  # Rotation matrices M (sample to crystal)
        self._R = None  # Rotation matrices R (crystal to sample)

        # keeps history of Figure, AxesSubplot and EBSDMap objects in these
        # lists. self.clear_history() can be used to clear the history
        self.figs = []
        self.axes = []
        self.ebsdmaps = []

    def __getitem__(self, colname):
        return self.data[colname].values

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
            self._R = euler_angles_to_rotation_matrix(self.phi1, self.Phi, self.phi2)
        return self._R

    def get_neighbors_oim(self, distance):
        """
        Returns list of relative indices of the neighboring pixels for
        a given distance in pixels
        """
        if self.grid.lower() == "hexgrid":
            R60 = np.array(
                [[self._cos60, -self._sin60], [self._sin60, self._cos60]]
            )  # 60 degrees rotation matrix

            j_list = np.arange(-distance, distance, 2)
            i_list = np.full(j_list.shape, -distance)

            xy = np.vstack([j_list * self._cos60, i_list * self._sin60])

            j_list, i_list = list(j_list), list(i_list)

            for r in range(1, 6):
                xy = np.dot(R60, xy)  # 60 degrees rotation
                j_list += list((xy[0] / self._cos60).round(0).astype(int))
                i_list += list((xy[1] / self._sin60).round(0).astype(int))
        else:  # sqrgrid
            R90 = np.array([[0, -1], [1, 0]], dtype=int)  # 90 degrees rotation matrix
            xy = np.vstack(
                [
                    np.arange(-distance, distance, dtype=int),
                    np.full(2 * distance, -distance, dtype=int),
                ]
            )

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
        if self.grid.lower() == "hexgrid":
            if distance > self._n_neighbors_hexgrid_fixed:
                raise Exception(
                    "get_neighbors_fixed not supported for distance > {}".format(
                        self._n_neighbors_hexgrid_fixed
                    )
                )
            j_list, i_list = list(zip(*self.neighbors_hexgrid_fixed[distance - 1]))
        else:
            raise Exception(
                "get_neighbors_fixed not yet supported for grid type {}".format(
                    self.grid
                )
            )
        return list(j_list), list(i_list)

    def get_neighbors(
        self, distance, perimeteronly=True, distance_convention="OIM", sel=None
    ):
        """
        Returns list of indices of the neighboring pixels for every pixel
        for a given distance in pixels

        Arguments
        ---------
        distance : int
            Distance with respect to the central pixel defined in terms of
            the nearest neighbor, i.e., distance = 3 represents the 3rd
            closest neighbor pixels
        perimeteronly : bool (optional)
            If True, considers only pixels in the perimeter. If False, then
            also includes innermost pixels
            Default : True
        distance_convention : str (optional)
            Distance convention used for selecting the neighboring pixels.
            Two possible values are allowed: 'OIM' or 'fixed'.
            The OIM convention is used by the TSL OIM software and is
            explained in its manual. 'fixed' stands for fixed distance,
            meaning that the neighbors are defined based on a fixed
            distance from the central pixel.
            Default : OIM
        sel : bool numpy 1D array (optional)
            Boolean array indicating data points calculations should be
            performed
            Default: None

        Returns
        -------
        neighbors_ind : numpy ndarray shape(N, K) - K being the number of
            neighbors
            Indices of the neighboring pixels
        """
        if distance_convention.lower() == "oim":
            _get_neighbors = self.get_neighbors_oim
        elif distance_convention.lower() == "fixed":
            _get_neighbors = self.get_neighbors_fixed
        else:
            raise Exception(
                'get_neighbors: unknown distance convention "{}"'.format(
                    distance_convention
                )
            )

        if perimeteronly:
            # only pixels in the perimeter
            j_shift, i_shift = _get_neighbors(distance)
        else:
            # including inner pixels
            j_shift, i_shift = [], []
            for d in range(1, distance + 1):
                j_sh, i_sh = _get_neighbors(d)
                j_shift += j_sh
                i_shift += i_sh

        n_neighbors = len(j_shift)
        if sel is None:
            sel = np.full(self.N, True, dtype=bool)

        # x
        j_neighbors = np.full((self.N, n_neighbors), -1, dtype=int)
        j_neighbors[sel] = np.add.outer(self.j[sel], j_shift)
        # y
        i_neighbors = np.full((self.N, n_neighbors), -1, dtype=int)
        i_neighbors[sel] = np.add.outer(self.i[sel], i_shift)

        # i, j out of allowed range
        outliers = (
            (j_neighbors < 0)
            | (j_neighbors >= self.ncols)
            | (i_neighbors < 0)
            | (i_neighbors >= self.nrows)
        )

        neighbors_ind = np.full((self.N, n_neighbors), -1, dtype=int)
        neighbors_ind[sel] = self.ij_to_index(i_neighbors[sel], j_neighbors[sel])
        neighbors_ind[outliers] = -1

        return neighbors_ind.astype(int)

    def get_distance_neighbors(self, distance, distance_convention="OIM"):
        """
        Returns distance, in um, to the n-th (distance-th) neighbor

        Arguments
        ---------
        distance : int
            Distance with respect to the central pixel defined in terms of
            the nearest neighbor, i.e., distance = 3 represents the 3rd
            closest neighbor pixels
        distance_convention : str (optional)
            Distance convention used for selecting the neighboring pixels.
            Two possible values are allowed: 'OIM' or 'fixed'.
            The OIM convention is used by the TSL OIM software and is
            explained in its manual. 'fixed' stands for fixed distance,
            meaning that the neighbors are defined based on a fixed
            distance from the central pixel.
            Default : OIM

        Returns
        -------
        d : float
            Distance, in um, to the n-th (distance-th) neighbor
        """
        if distance_convention.lower() == "oim":
            j, i = self.get_neighbors_oim(distance)
        elif distance_convention.lower() == "fixed":
            j, i = self.get_neighbors_fixed(distance)
        else:
            raise Exception(
                ("get_distance_neighbors: unknown distance convention " '"{}"').format(
                    distance_convention
                )
            )

        if self.grid.lower() == "hexgrid":
            d = 0.5 * (np.array(j) ** 2 + 3.0 * np.array(i) ** 2) ** 0.5
        else:  # sqrgrid
            d = (np.array(j) ** 2 + np.array(i) ** 2) ** 0.5

        return d.mean()

    def get_KAM(
        self,
        distance=1,
        perimeteronly=True,
        maxmis=None,
        distance_convention="OIM",
        sel=None,
        **kwargs
    ):
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
        KAM : numpy ndarray shape(N) with KAM values in degrees
        """
        neighbors = self.get_neighbors(
            distance, perimeteronly, distance_convention, sel
        )
        return kernel_average_misorientation(
            self.M, neighbors, sel, maxmis, kwargs.pop("out", "deg"), **kwargs
        )

    def plot_IPF(
        self,
        d=[0, 0, 1],
        ax=None,
        sel=None,
        colorfill="black",
        gray=None,
        graymin=0,
        graymax=None,
        tiling=None,
        w=2048,
        scalebar=True,
        plotlimits=None,
        verbose=True,
        **kwargs
    ):
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
        colorfill : str or list shape(3) or shape(4) (optional)
            Color used to fill unindexed pixels. It can be provided as RGB
            or RGBA values as an iterable. If RGBA is provided, alpha channel
            is droppped
            Default: 'black'
        gray : numpy ndarray (optional)
            Grayscale mask plotted over IPF.
            For example, one may want to overlay the IPF map with the image
            quality data.
            Default: None
        graymin : float (optional)
            Minimum gray value used for calculation of the gray mask.
            If None, min(gray) is used.
            Default: 0
        graymax : float (optional)
            Maximum gray value used for calculation of the gray mask.
            If None, max(gray) is used.
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
        plotlimits: tuple, list, ndarray, EBSDMap or AxesSubplot object
            (optional)
            x and y limits of the plot. It works similarly as sel. Only
            pixels fall inside the provided x and y limits are plotted.
            If an EBSDMap or AxesSubplot object is provided, then limits
            are retrieved by calling get_xlim() and get_ylim() functions.
            Default: None
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
        xlim, ylim = None, None
        if isinstance(plotlimits, (EBSDMap, Axes)):
            xlim, ylim = plotlimits.get_xlim(), plotlimits.get_ylim()
        elif isinstance(plotlimits, (tuple, list, np.ndarray)):
            if len(plotlimits) == 4:
                xlim, ylim = plotlimits[:2], plotlimits[2:]
            else:
                print("plotlimits should be provided as list/tuple of length 4")
        if xlim is not None and ylim is not None:
            xlim, ylim = sorted(xlim), sorted(ylim)
            sellim = (
                (self.x >= xlim[0])
                & (self.x <= xlim[1])
                & (self.y >= ylim[0])
                & (self.y <= ylim[1])
            )
            if sel is None:
                sel = sellim
            else:
                sel = sel & sellim

        if isinstance(gray, str):
            try:
                gray = getattr(self, gray)
            except AttributeError:
                gray = self.data[gray].values

        ebsdmap = plot_IPF(
            self.M,
            self.nrows,
            self.ncols_odd,
            self.ncols_even,
            self.x,
            self.y,
            self.grid,
            self.dx,
            self.dy,
            d,
            ax,
            sel,
            colorfill,
            gray,
            graymin,
            graymax,
            tiling,
            w,
            scalebar,
            verbose,
            **kwargs
        )
        self.ebsdmaps.append(ebsdmap)
        self.figs.append(ebsdmap.fig)
        self.axes.append(ebsdmap.ax)
        return ebsdmap

    def plot_property(
        self,
        prop,
        propname="z",
        ax=None,
        colordict=None,
        colorfill="black",
        fillvalue=np.nan,
        sel=None,
        gray=None,
        graymin=0,
        graymax=None,
        tiling=None,
        w=2048,
        scalebar=True,
        colorbar=True,
        plotlimits=None,
        verbose=True,
        **kwargs
    ):
        """
        Plots any EBSD property

        Parameters
        ----------
        prop : array shape(N)
            Property to be plotted provided as np.ndarray(N), where N is the
            size of the data file
        propname : str
            Property name (optional)
            Default: value
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
        graymin : float (optional)
            Minimum gray value used for calculation of the gray mask.
            If None, min(gray) is used.
            Default: 0
        graymax : float (optional)
            Maximum gray value used for calculation of the gray mask.
            If None, max(gray) is used.
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
        plotlimits: tuple, list, ndarray, EBSDMap or AxesSubplot object
            (optional)
            x and y limits of the plot. It works similarly as sel. Only
            pixels fall inside the provided x and y limits are plotted.
            If an EBSDMap or AxesSubplot object is provided, then limits
            are retrieved by calling get_xlim() and get_ylim() functions.
            Default: None
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
        xlim, ylim = None, None
        if isinstance(plotlimits, (EBSDMap, Axes)):
            xlim, ylim = plotlimits.get_xlim(), plotlimits.get_ylim()
        elif isinstance(plotlimits, (tuple, list, np.ndarray)):
            if len(plotlimits) == 4:
                xlim, ylim = plotlimits[:2], plotlimits[2:]
            else:
                print("plotlimits should be provided as list/tuple of length 4")
        if xlim is not None and ylim is not None:
            xlim, ylim = sorted(xlim), sorted(ylim)
            sellim = (
                (self.x >= xlim[0])
                & (self.x <= xlim[1])
                & (self.y >= ylim[0])
                & (self.y <= ylim[1])
            )
            if sel is None:
                sel = sellim
            else:
                sel = sel & sellim

        if isinstance(prop, str):
            try:
                prop = getattr(self, prop)
            except AttributeError:
                prop = self.data[prop].values

        if isinstance(gray, str):
            try:
                gray = getattr(self, gray)
            except AttributeError:
                gray = self.data[gray].values

        ebsdmap = plot_property(
            prop,
            self.nrows,
            self.ncols_odd,
            self.ncols_even,
            self.x,
            self.y,
            self.grid,
            self.dx,
            self.dy,
            propname,
            ax,
            colordict,
            colorfill,
            fillvalue,
            sel,
            gray,
            graymin,
            graymax,
            tiling,
            w,
            scalebar,
            colorbar,
            verbose,
            **kwargs
        )
        self.ebsdmaps.append(ebsdmap)
        self.figs.append(ebsdmap.fig)
        self.axes.append(ebsdmap.ax)
        return ebsdmap

    def plot_phase(
        self,
        ax=None,
        colordict=None,
        colorfill="black",
        fillvalue=-1,
        sel=None,
        gray=None,
        graymin=0,
        graymax=None,
        tiling=None,
        w=2048,
        scalebar=True,
        plotlimits=None,
        verbose=True,
        **kwargs
    ):
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
        graymin : float (optional)
            Minimum gray value used for calculation of the gray mask.
            If None, min(gray) is used.
            Default: 0
        graymax : float (optional)
            Maximum gray value used for calculation of the gray mask.
            If None, max(gray) is used.
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
        plotlimits: tuple, list, ndarray, EBSDMap or AxesSubplot object
            (optional)
            x and y limits of the plot. It works similarly as sel. Only
            pixels fall inside the provided x and y limits are plotted.
            If an EBSDMap or AxesSubplot object is provided, then limits
            are retrieved by calling get_xlim() and get_ylim() functions.
            Default: None
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
        if colordict is None:
            ph_code = set(self.ph)
            ccycler = cycle(self.colors)
            colordict = {ph: next(ccycler) for ph in ph_code}

        ebsdmap = self.plot_property(
            self.ph,
            "phase",
            ax,
            colordict,
            colorfill,
            fillvalue,
            sel,
            gray,
            graymin,
            graymax,
            tiling,
            w,
            scalebar,
            False,
            plotlimits,
            verbose,
            **kwargs
        )
        return ebsdmap

    def plot_KAM(
        self,
        distance=1,
        perimeteronly=True,
        ax=None,
        maxmis=None,
        distance_convention="OIM",
        colorfill="black",
        fillvalue=np.nan,
        sel=None,
        gray=None,
        graymin=0,
        graymax=None,
        tiling=None,
        w=2048,
        scalebar=True,
        colorbar=True,
        plotlimits=None,
        verbose=True,
        **kwargs
    ):
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
        graymin : float (optional)
            Minimum gray value used for calculation of the gray mask.
            If None, min(gray) is used.
            Default: 0
        graymax : float (optional)
            Maximum gray value used for calculation of the gray mask.
            If None, max(gray) is used.
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
        plotlimits: tuple, list, ndarray, EBSDMap or AxesSubplot object
            (optional)
            x and y limits of the plot. It works similarly as sel. Only
            pixels fall inside the provided x and y limits are plotted.
            If an EBSDMap or AxesSubplot object is provided, then limits
            are retrieved by calling get_xlim() and get_ylim() functions.
            Default: None
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
        xlim, ylim = None, None
        if isinstance(plotlimits, (EBSDMap, Axes)):
            xlim, ylim = plotlimits.get_xlim(), plotlimits.get_ylim()
        elif isinstance(plotlimits, (tuple, list, np.ndarray)):
            if len(plotlimits) == 4:
                xlim, ylim = plotlimits[:2], plotlimits[2:]
            else:
                print("plotlimits should be provided as list/tuple of length 4")
        if xlim is not None and ylim is not None:
            xlim, ylim = sorted(xlim), sorted(ylim)
            sellim = (
                (self.x >= xlim[0])
                & (self.x <= xlim[1])
                & (self.y >= ylim[0])
                & (self.y <= ylim[1])
            )
            if sel is None:
                sel = sellim
            else:
                sel = sel & sellim

        KAM = self.get_KAM(distance, perimeteronly, maxmis, distance_convention, sel)
        ebsdmap = self.plot_property(
            KAM,
            "KAM",
            ax,
            None,
            colorfill,
            fillvalue,
            sel,
            gray,
            graymin,
            graymax,
            tiling,
            w,
            scalebar,
            colorbar,
            None,
            verbose,
            **kwargs
        )
        ebsdmap.cax.set_label("KAM (°)")
        return ebsdmap

    def plot_PF(
        self,
        proj=[1, 0, 0],
        ax=None,
        sel=None,
        rotation=None,
        contour=False,
        sep_phases=False,
        verbose=True,
        **kwargs
    ):
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
        sep_phases : bool (optional)
            If true, plot pole figures of different phases separately
            Default: False
        verbose : bool (optional)
            If True, prints computation time
            Default: True

        **kwargs :
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

        # TO DO: SET DIFFERENT COLOR SCHEMES FOR EACH PLOT
        if sep_phases:
            phases = np.unique(self.ph[sel])
            ph = phases[0]
            ax = plot_PF(
                None,
                proj,
                ax,
                sel & (self.ph == ph),
                rotation,
                contour,
                verbose,
                R=self.R,
                label="{}".format(ph),
                **kwargs
            )
            for ph in phases[1:]:
                plot_PF(
                    None,
                    proj,
                    ax,
                    sel & (self.ph == ph),
                    rotation,
                    contour,
                    verbose,
                    R=self.R,
                    ax=ax,
                    label="{}".format(ph),
                    **kwargs
                )

        else:
            ax = plot_PF(
                None, proj, ax, sel, rotation, contour, verbose, R=self.R, **kwargs
            )

        return ax

    def savefig(self, fname, **kwargs):
        """
        Saves EBSD map plotted last

        Parameters
        ----------
        fname : str
            File name

        **kwargs :
            kwargs parameters are passed to ebsdmap.fig.savefig(fname, **kwargs)
            function
        """
        self.ebsdmaps[-1].savefig(fname, **kwargs)

    def save_ang_file(self, fname, sel=None, **kwargs):
        """
        Export ScanData as ang file

        Parameters
        ----------
        fname : str
            File name
        sel : list of array of booleans
            selection
        """
        if sel is None:
            newscan = self
        else:
            newscan = selection_to_scandata(self, sel)

        header = []
        if isinstance(newscan.header, list):
            header = newscan.header
            for i, line in enumerate(header):
                if "# NCOLS_ODD:" in line:
                    header[i] = "# NCOLS_ODD: {:d}\n".format(newscan.ncols_odd)
                    continue
                if "# NCOLS_EVEN:" in line:
                    header[i] = "# NCOLS_EVEN: {:d}\n".format(newscan.ncols_even)
                    continue
                if "# NROWS:" in line:
                    header[i] = "# NROWS: {:d}\n".format(newscan.nrows)
                    continue
        else:
            header.append("# GRID: {:}\n".format(newscan.grid))
            header.append("# XSTEP: {:.8f}\n".format(newscan.dx))
            header.append("# YSTEP: {:.8f}\n".format(newscan.dy))
            header.append("# NCOLS_ODD: {:d}\n".format(newscan.ncols_odd))
            header.append("# NCOLS_EVEN: {:d}\n".format(newscan.ncols_even))
            header.append("# NROWS: {:d}\n".format(newscan.nrows))

        try:
            file = open(fname, "w")
            file.write("".join(header))
            file.close()
            newscan.data.to_csv(
                fname,
                mode="a",
                header=False,
                index=False,
                sep=" ",
                float_format=kwargs.pop("float_format", "%.5f"),
            )
        except Exception:
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

    xmin = scan.dx * (2.0 * jmin - 1.0) / 4.0  # (jmin*dx/2 - dx/4)
    xmax = scan.dx * (2.0 * jmax + 1.0) / 4.0  # (jmax*dx/2 + dx/4)
    ymin = scan.dy * (2.0 * imin - 1.0) / 2.0  # (imin*dy - dy/2)
    ymax = scan.dy * (2.0 * imax + 1.0) / 2.0  # (imax*dy + dy/2)

    ncols_even = (jmax - jmin) // 2
    ncols_odd = ncols_even + 1
    nrows = imax - imin + 1

    # select rectangle surrounding the selected data
    rect = (scan.x >= xmin) & (scan.x <= xmax) & (scan.y >= ymin) & (scan.y <= ymax)

    # total number of points
    N = ncols_even * (nrows // 2) + ncols_odd * (nrows - nrows // 2)
    if N != np.count_nonzero(rect):
        raise Exception(
            (
                "Something went wrong: expected number "
                "of points ({}) differs from what "
                "we got ({})"
            ).format(N, np.count_nonzero(rect))
        )

    return ncols_odd, ncols_even, nrows, rect


def _get_rectangle_surrounding_selection_sqrgrid(scan, sel):
    """
    Select rectangle surrounding the selected data.
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

    xmin = scan.x[ind_xmin]
    xmax = scan.x[ind_xmax]
    ymin = scan.y[ind_ymin]
    ymax = scan.y[ind_ymax]

    ncols_even = jmax - jmin + 1
    ncols_odd = ncols_even
    nrows = imax - imin + 1

    # select rectangle surrounding the selected data
    rect = (scan.x >= xmin) & (scan.x <= xmax) & (scan.y >= ymin) & (scan.y <= ymax)

    # total number of points
    N = ncols_even * nrows
    if N != np.count_nonzero(rect):
        raise Exception(
            (
                "Something went wrong: expected number "
                "of points ({}) differs from what "
                "we got ({})"
            ).format(N, np.count_nonzero(rect))
        )

    return ncols_odd, ncols_even, nrows, rect


def _get_rectangle_surrounding_selection(scan, sel):
    if scan.grid.lower() == "hexgrid":
        return _get_rectangle_surrounding_selection_hexgrid(scan, sel)
    else:
        return _get_rectangle_surrounding_selection_sqrgrid(scan, sel)


def selection_to_scandata(scan, sel):
    """
    Convert selection to new ScanData object

    Parameters
    ----------
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
    newdata.loc[~sel, "phi1"] = 4.0
    newdata.loc[~sel, "Phi"] = 4.0
    newdata.loc[~sel, "phi2"] = 4.0
    newdata.loc[~sel, "IQ"] = -1
    newdata.loc[~sel, "CI"] = -2
    newdata.loc[~sel, "ph"] = -1
    newdata.loc[~sel, "intensity"] = -1
    newdata.loc[~sel, "fit"] = 0

    ncols_odd, ncols_even, nrows, rect = _get_rectangle_surrounding_selection(scan, sel)

    # data to be exported is a rectangle
    newdata = newdata[rect]

    # offset x and y so (xmin, ymin) becomes the origin (0, 0)
    newdata.x -= newdata.x.min()
    newdata.y -= newdata.y.min()

    return ScanData(
        newdata, scan.grid, scan.dx, scan.dy, ncols_odd, ncols_even, nrows, scan.header
    )
