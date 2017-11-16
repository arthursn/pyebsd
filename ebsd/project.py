import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt 

from .orientation import euler_rotation
from .load_data import load_ang_file
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
    def __init__(self, fname):
        self.fname = fname
        raw = load_ang_file(fname)

        self.data = raw[0]
        self.grid = raw[1]
        self.dx = raw[2]
        self.dy = raw[3]
        self.ncols_odd = raw[4]
        self.ncols_even = raw[5]
        self.nrows = raw[6]
        self.ncols = self.ncols_odd + self.ncols_even
        self.N = len(self.data)

        self.ind = np.arange(self.N, dtype=int)
        self.phi1 = self.data[:,0]
        self.Phi = self.data[:,1]
        self.phi2 = self.data[:,2]
        self.x = self.data[:,3]
        self.y = self.data[:,4]
        self.IQ = self.data[:,5]
        self.CI = self.data[:,6]
        self.ph = self.data[:,7]

        self.R = euler_rotation(self.phi1, self.Phi, self.phi2)

        self.figs_maps = []
        self.axes_maps = []
    
    @property
    def M(self):
        return self.R.transpose([0,2,1])

    @property
    def i(self):
        return self.ind//self.ncols 

    @property
    def j(self):
        rem = self.ind%self.ncols
        return rem//self.ncols_odd + 2*(rem%self.ncols_odd)

    @property
    def neighbors(self, distance=0):
        """
        Returns list of indices of the neighboring pixels for each pixel
        """
        i0, j0 = self.i, self.j
        i1_, i1 = i0-1, i0+1
        j2_, j1_, j1, j2 = j0-2, j0-1, j0+1, j0+2

        i_near = np.ndarray((self.N, 6), dtype=int)
        j_near = np.vstack([j2_, j1_, j1, j2, j1, j1_]).T.astype(int)
        i_near[j0%2==0] = (np.vstack([i0, i1_, i1_, i0, i0, i0]).T)[j0%2==0]
        i_near[j0%2==1] = (np.vstack([i0, i0, i0, i0, i1, i1]).T)[j0%2==1]

        near = self.ij2ind(i_near, j_near)
        near[(near < 0) | (near >= self.N)] = -1
        return near.astype(int)
    
    def ij2ind(self, i, j):
        """
        i, j grid positions to index
        """
        # 1 - self.N*(j/self.ncols) turns negative every i, j pair where j extrapolates ncols
        return (1 - self.N*(j//self.ncols))*(i*self.ncols + (j%2)*self.ncols_odd + (j//2))

    def plot_IPF(self, d='ND', ax=None, sel=None, gray=None, tiling='rect', w=2048, scalebar=True, verbose=True, **kwargs):
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
        """
        ebsdmap = plot_IPF(self.R, self.nrows, self.ncols_even, self.ncols_odd, self.x, self.y, self.dx, d, ax, sel, gray, tiling, w, scalebar, verbose, **kwargs)
        self.figs_maps.append(ebsdmap.ax.get_figure())
        self.axes_maps.append(ebsdmap.ax)
        return ebsdmap

    def plot_property(self, prop, ax=None, colordict=None, colorfill=[0,0,0,1], sel=None, gray=None, tiling='rect', w=2048, scalebar=True, verbose=True, **kwargs):
        ebsdmap = plot_property(prop, self.nrows, self.ncols_even, self.ncols_odd, self.x, self.y, self.dx, ax, colordict, colorfill, sel, gray, tiling, w, scalebar, verbose, **kwargs)
        self.figs_maps.append(ebsdmap.ax.get_figure())
        self.axes_maps.append(ebsdmap.ax)
        return ebsdmap

    def plot_phase(self, ax=None, colordict={'1': [1,0,0,1], '2': [0,1,0,1]}, colorfill=[0,0,0,1], sel=None, gray=None, tiling='rect', w=2048, scalebar=True, verbose=True, **kwargs):
        ebsdmap = self.plot_property(self.ph, ax, colordict, colorfill, sel, gray, tiling, w, scalebar, verbose, **kwargs)
        self.figs_maps.append(ebsdmap.ax.get_figure())
        self.axes_maps.append(ebsdmap.ax)
        return ebsdmap

    def plot_PF(self, proj=[1,0,0], ax=None, sel=None, parent_or=None, contour=False, verbose=True, **kwargs):
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
        """
        return plot_PF(self.R, None, proj, ax, sel, parent_or, contour, verbose, **kwargs)

    def savefig(self, fname, **kwargs):
        kwargs.update({'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.0})
        plt.savefig(fname, **kwargs)

def load_scandata(fname):
    return Scandata(fname)