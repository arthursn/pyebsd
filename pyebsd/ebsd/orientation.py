import sys
import time
import numpy as np

__all__ = ['trace_to_angle', 'stereographic_projection',
           'stereographic_projection_to_direction', 'average_orientation',
           'misorientation', 'misorientation_neighbors',
           'kernel_average_misorientation', 'minimize_disorientation',
           'euler_angles_to_rotation_matrix', 'rotation_matrix_to_euler_angles',
           'axis_angle_to_rotation_matrix', 'list_cubic_symmetry_operators_KS',
           'list_cubic_symmetry_operators', 'list_cubic_family_directions',
           'reduce_cubic_transformations', 'IPF', 'PF']


def trace_to_angle(tr, out='deg'):
    """
    Converts the trace of a orientation matrix to the misorientation angle
    """
    ang = np.arccos((tr-1.)/2.)
    if out == 'deg':
        ang = np.degrees(ang)
    return ang


""" Steographic projection functions """


def stereographic_projection(d, norm=True, coord='cartesian'):
    """
    Returns the coordinates of the stereographic projection of a direction 
    'd'

    Arguments
    ---------
    d : 1D or 2D list or array shape (3) or (,3)
        Direction
    """
    d = np.asarray(d)
    ndim = d.ndim
    shp = d.shape

    if ndim == 1:
        d = d.reshape(3, 1)
    elif ndim == 2:
        # transpose d
        d = d.T

    if norm:
        d = d/np.linalg.norm(d, axis=0)

    c0 = d[0]/(1. + d[2])
    c1 = d[1]/(1. + d[2])

    if coord == 'polar':
        r = (c0**2. + c1**2.)**.5
        theta = np.arctan2(c1, c0)
        theta[theta < 0] = theta[theta < 0] + 2*np.pi
        c0, c1 = r, theta

    if ndim == 1:
        c0, c1 = np.float(c0), np.float(c1)

    return c0, c1


def stereographic_projection_to_direction(xy):
    """
    Calculates the direction from the stereographic projection coordinates

    Arguments
    ---------
    xy : iterable shape(2) or shape(N, 2)
        List with stereographic projection coordinates

    Returns
    -------
    uvw : np.ndarray(3) or np.ndarray(N, 3)
        List with corresponding directions
    """
    xy = np.asarray(xy)
    ndim = xy.ndim

    if ndim == 1:
        xy = xy.reshape(1, -1)

    x = xy[:, 0]
    y = xy[:, 1]

    uvw = np.ndarray((len(xy), 3))

    uvw[:, 2] = (1. - x**2. - y**2.)/(1. + x**2. + y**2.)
    uvw[:, 0] = x*(1. + uvw[:, 2])
    uvw[:, 1] = y*(1. + uvw[:, 2])

    if ndim == 1:
        uvw = uvw.ravel()

    return uvw


""" Misorientation functions """


def average_orientation(M, sel=None, **kwargs):
    """
    Calculates rotation matrix corresponding to average orientation

    M : numpy ndarray shape(N, 3, 3)
        List of rotation matrices describing the rotation from the sample 
        coordinate frame to the crystal coordinate frame
    sel : bool numpy 1D array (optional)
        Boolean array indicating data points calculations should be 
        performed
        Default: None
    **kwargs :
        verbose : bool (optional)
            If True, prints computation time
            Default: True
        vectorized : bool (optional)
            If True, performs all operations vectorized using numpy
            Default: True

    Returns
    -------
    M_avg : numpy ndarray shape(3, 3)
        Average orientation matrix
    """
    # verbose is pased to 'rotation_matrix_to_euler_angles', so use kwargs.get, not kwargs.pop
    verbose = kwargs.get('verbose', True)
    if verbose:
        t0 = time.time()
        sys.stdout.write('Calculating average orientation... ')
        sys.stdout.flush()

    if isinstance(sel, np.ndarray):
        M_sel = M[sel]
    else:
        M_sel = M

    N = len(M_sel)
    MrefT = M_sel[N//2].T
    C = list_cubic_symmetry_operators()

    # 'vectorized' is passed to rotation_matrix_to_euler_angles, which in turn is passed
    # to minimize_disorientation. That's why I'm using kwargs.get instead of kwargs.pop
    if kwargs.get('vectorized', True):
        # 4 dimensional numpy narray(N,24,3,3)
        Mprime = np.tensordot(C, M_sel,
                              axes=[[-1], [-2]]).transpose([2, 0, 1, 3])
        # misorientation matrices D
        D = np.tensordot(Mprime, MrefT, axes=[[-1], [-2]])
        tr = np.trace(D, axis1=2, axis2=3)
        neg = tr < -1.
        tr[neg] = -tr[neg]
        Mprime[neg] = -Mprime[neg]
        M_sel = Mprime[(list(range(N)), np.argmax(tr, axis=1))]
    else:
        for i in range(N):
            Mprime = np.tensordot(C, M_sel[i], axes=[[-1], [-2]])
            D = np.tensordot(Mprime, MrefT, axes=[[-1], [-2]])
            tr = np.trace(D, axis1=1, axis2=2)
            neg = tr < 0.  # select negative traces
            tr[neg] = -tr[neg]
            Mprime[neg] = -Mprime[neg]
            M_sel[i] = Mprime[np.argmax(tr)]

    R_sel = M_sel.transpose([0, 2, 1])
    phi1, Phi, phi2 = rotation_matrix_to_euler_angles(R_sel, avg=True, **kwargs)

    M_avg = euler_angles_to_rotation_matrix(phi1, Phi, phi2, verbose=False).T

    if verbose:
        sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))
        sys.stdout.flush()

    del D, Mprime, M_sel, R_sel
    return M_avg


def misorientation(A, B, out='deg'):
    """
    Calculates the misorientation between A e B

    Parameters
    ----------
    A : numpy ndarray shape(3, 3)
        First rotation matrix
    B : numpy ndarray shape(3, 3)
        Second rotation matrix
    out : str (optional)
        Unit of the output. Possible values are:
        'tr': as a trace value of the misorientation matrix
        'deg': as misorientation angle in degrees
        'rad': as misorientation angle in radians
        Default: 'deg'

    Returns
    -------
    misang : float
        Misorientation angle given in the unit specified by 'out'
    """

    C = list_cubic_symmetry_operators()
    Adim, Bdim = np.ndim(A), np.ndim(B)

    if (Adim == 2) and (Bdim == 2):
        # Tj = Cj * A * B^T
        T = np.tensordot(C, A.dot(B.T), axes=[[-1], [-2]])
        tr = T.trace(axis1=1, axis2=2)
        x = tr.max()  # Maximum trace
        # This might happen due to rounding error
        if x > 3.:
            x = 3.
        if out != 'tr':
            x = np.arccos((x-1.)/2.)  # mis x in radians
            if out == 'deg':
                x = np.degrees(x)  # mis x in degrees
    else:
        raise Exception('Invalid shapes of arrays A or B')
    return x


def misorientation_neighbors(M, neighbors, sel=None, out='deg', **kwargs):
    """
    Calculates the misorientation angle of every data point with respective
    orientation matrix provided in 'M' with respect to an arbitrary number 
    of neighbors, whose indices are provided in the 'neighbors' argument.

    Parameters
    ----------
    M : numpy ndarray shape(N, 3, 3)
        List of rotation matrices describing the rotation from the sample 
        coordinate frame to the crystal coordinate frame
    neighbors : numpy ndarray shape(N, K) - K being the number of neighbors
        Indices of the neighboring pixels
    sel : bool numpy 1D array (optional)
        Boolean array indicating data points calculations should be 
        performed
        Default: None
    out : str (optional)
        Unit of the output. Possible values are:
        'deg': angle(s) in degrees
        'rad': angle(s) in radians
        Default: 'deg'
    **kwargs :
        verbose : bool (optional)
            If True, prints computation time
            Default: True

    Returns
    -------
    misang : numpy ndarray shape(N, K) - K being the number of neighbors
        KAM : numpy ndarray shape(N) with KAM values
    """
    N = M.shape[0]
    nneighbors = neighbors.shape[1]

    C = list_cubic_symmetry_operators()

    # 2D array to store trace values initialized as -2 (trace values are
    # always in the [-1, 3] interval)
    tr = np.full((N, nneighbors), -2., dtype=float)
    # 2D array to store the misorientation angles in degrees
    misang = np.full((N, nneighbors), -1., dtype=float)

    if not isinstance(sel, np.ndarray):
        sel = np.full(N, True, dtype=bool)

    verbose = kwargs.pop('verbose', True)
    if verbose:
        t0 = time.time()
        sys.stdout.write('Calculating misorientations for {} points for {} neighbors'.format(
            np.count_nonzero(sel), nneighbors))
        sys.stdout.write(' [')
        sys.stdout.flush()

    for k in range(nneighbors):
        # valid points, i.e., those part of the selection and with valid neighrbor index (> 0)
        ok = (neighbors[:, k] >= 0) & sel & sel[neighbors[:, k]]
        # Rotation from M[ok] to M[neighbors[ok, k]]
        # Equivalent to np.matmul(M[neighbors[ok,k]], M[ok].transpose([0,2,1]))
        T = np.einsum('ijk,imk->ijm', M[neighbors[ok, k]], M[ok])

        for m in range(len(C)):
            # Smart way to calculate the trace using einsum.
            # Equivalent to np.matmul(C[m], T).trace(axis1=1, axis2=2)
            a, b = C[m].nonzero()
            ttr = np.einsum('j,ij->i', C[m, a, b], T[:, a, b])
            tr[ok, k] = np.max(np.vstack([tr[ok, k], ttr]), axis=0)

        if verbose:
            if k > 0 and k < nneighbors:
                sys.stdout.write(', ')
            sys.stdout.write('{}'.format(k + 1))
            sys.stdout.flush()

    del T, ttr

    if verbose:
        sys.stdout.write('] in {:.2f} s\n'.format(time.time() - t0))
        sys.stdout.flush()

    # Take care of tr > 3. that might happend due to rounding errors
    tr[tr > 3.] = 3.

    # Filter out invalid trace values
    ok = tr >= -1.
    misang[ok] = trace_to_angle(tr[ok], out)
    return misang


def kernel_average_misorientation(M, neighbors, sel=None, maxmis=None, out='deg', **kwargs):
    """
    Calculates the Kernel Average Misorientation (KAM)

    M : numpy ndarray shape(N, 3, 3)
        List of rotation matrices describing the rotation from the sample 
        coordinate frame to the crystal coordinate frame
    neighbors : numpy ndarray shape(N, K) - K being the number of neighbors
        Indices of the neighboring pixels
    sel : bool numpy 1D array (optional)
        Boolean array indicating data points calculations should be 
        performed
        Default: None
    out : str (optional)
        Unit of the output. Possible values are:
        'deg': angle(s) in degrees
        'rad': angle(s) in radians
        Default: 'deg'
    **kwargs :
        verbose : bool (optional)
            If True, prints computation time
            Default: True

    Returns
    -------
    KAM : numpy ndarray shape(N) - M being the number of neighbors
        KAM : numpy ndarray shape(N) with KAM values
    """
    misang = misorientation_neighbors(M, neighbors, sel, out, **kwargs)

    outliers = misang < 0  # filter out negative values
    if maxmis is not None:
        outliers |= misang > maxmis  # and values > maxmis

    misang[outliers] = 0.
    nneighbors = np.count_nonzero(~outliers, axis=1)

    noneighbors = nneighbors == 0
    nneighbors[noneighbors] = 1  # to prevent division by 0

    KAM = np.sum(misang, axis=1)/nneighbors
    KAM[noneighbors] = np.nan  # invalid KAM when nneighbors is 0

    return KAM


def minimize_disorientation(V, V0, **kwargs):
    """
    Calculates the orientation that truly minimizes the disorientation
    between the list of orientations V and a single orientation V0.
    """
    n = kwargs.pop('n', 5)  # grid size
    maxdev = kwargs.pop('maxdev', .25)  # maximum deviation in degrees
    maxdev = np.radians(maxdev)  # maxdev in radians
    it = kwargs.pop('it', 3)  # number of iterations
    verbose = kwargs.pop('verbose', False)
    plot = kwargs.pop('plot', False)
    if verbose:
        sys.stdout.write('\nMinimizing disorientation...\n')
        sys.stdout.flush()
    if plot:
        from .plotting import plot_PF
        axmin = plot_PF(M=V, ms=.5, c='k', verbose=False)
    for i in range(it):
        step = maxdev/n
        t = np.linspace(-maxdev, maxdev, n)
        # Euler space in XYZ convention
        theta, phi, psi = np.meshgrid(t, t, t)
        theta, phi, psi = theta.ravel(), phi.ravel(), psi.ravel()
        # Rotation matrices from euler angles shape(n^3, 3, 3)
        A = euler_angles_to_rotation_matrix(theta, phi, psi, conv='xyz', verbose=False)
        # Rotate V0 by A. Resulting B is shape(n^3, 3, 3)
        B = np.tensordot(A, V0, axes=[[-1], [-2]])
        # Calculate rotation from B to V. Resulting D is shape(len(V), n^3, 3, 3)
        D = np.tensordot(V, B.transpose(0, 2, 1), axes=[[-1], [-2]]).transpose([0, 2, 1, 3])
        # Average (mean) trace of D along axis 0 shape(n^3)
        tr = np.abs(np.trace(D, axis1=2, axis2=3)).mean(axis=0)
        # Index of maximum trace value
        imax = np.argmax(tr)
        if verbose:
            dth = np.degrees(np.arccos((np.trace(A[imax])-1.)/2.))
            sys.stdout.write('{:2d} : {:g}, {:g}, {:g}; mis = {:g} deg\n'.format(
                i+1, np.degrees(theta[imax]), np.degrees(phi[imax]), np.degrees(psi[imax]), dth))
        if plot:
            Rscan = A.dot(V0).transpose([0, 2, 1])
            plot_PF(R=Rscan, ax=axmin, scatter=True, s=30,
                    marker='s', c=np.repeat(tr, 3), verbose=False)
            plot_PF(R=Rscan[imax], ax=axmin, scatter=True, s=200, marker='x', c='r', verbose=False)

        V0 = A[imax].dot(V0)
        maxdev /= n
    del A, B, D, tr

    return V0


""" Rotation representation conversion functions """


def euler_angles_to_rotation_matrix(phi1, Phi, phi2, conv='zxz', **kwargs):
    """
    Given 3 Euler angles, calculates rotation R (active description).
    Please notice that the Euler angles in the ang files follow passive
    description.

    Parameters:
    -----------
    phi1 : float or list, tuple, or array(N)
    Phi : float or list, tuple, or array(N)
    phi2 : float or list, tuple, or array(N)
        Euler angles

    conv : string (optional)
        Rotation convention
        Default: zxz (Bunge notation)

    **kwargs :
        verbose : boolean
            If True (default), print calculation time

    """
    verbose = kwargs.pop('verbose', True)
    if verbose:
        t0 = time.time()
        sys.stdout.write('Calculating rotation matrices... ')
        sys.stdout.flush()

    if np.ndim(phi1) == 0:
        N = 1
    else:
        phi1 = np.asarray(phi1)
        Phi = np.asarray(Phi)
        phi2 = np.asarray(phi2)
        if len(phi1) == len(Phi) and len(phi1) == len(phi2):
            N = len(phi1)
        else:
            raise Exception('Lengths of phi1, Phi, and phi2 differ')

    cphi1, sphi1 = np.cos(phi1), np.sin(phi1)
    cPhi, sPhi = np.cos(Phi), np.sin(Phi)
    cphi2, sphi2 = np.cos(phi2), np.sin(phi2)
    R = np.ndarray((N, 3, 3))

    conv = conv.lower()
    if conv == 'zxz':
        R[:, 0, 0] = cphi1*cphi2 - sphi1*cPhi*sphi2
        R[:, 0, 1] = -cphi1*sphi2 - sphi1*cPhi*cphi2
        R[:, 0, 2] = sphi1*sPhi
        R[:, 1, 0] = sphi1*cphi2 + cphi1*cPhi*sphi2
        R[:, 1, 1] = -sphi1*sphi2 + cphi1*cPhi*cphi2
        R[:, 1, 2] = -cphi1*sPhi
        R[:, 2, 0] = sPhi*sphi2
        R[:, 2, 1] = sPhi*cphi2
        R[:, 2, 2] = cPhi
    elif conv == 'xyz':
        R[:, 0, 0] = cPhi*cphi1
        R[:, 0, 1] = -cPhi*sphi1
        R[:, 0, 2] = sPhi
        R[:, 1, 0] = cphi2*sphi1 + sphi2*sPhi*cphi1
        R[:, 1, 1] = cphi2*cphi1 - sphi2*sPhi*sphi1
        R[:, 1, 2] = -sphi2*cPhi
        R[:, 2, 0] = sphi2*sphi1 - cphi2*sPhi*cphi1
        R[:, 2, 1] = sphi2*cphi1 + cphi2*sPhi*sphi1
        R[:, 2, 2] = cphi2*cPhi
    else:
        raise Exception('"{}" convention not supported'.format(conv))

    if np.ndim(phi1) == 0:
        R = R.reshape(3, 3)

    if verbose:
        sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))
        sys.stdout.flush()

    return R


def rotation_matrix_to_euler_angles(R, conv='zxz', **kwargs):
    """
    Calculates the Euler angles from a rotation matrix or a sequence
    of rotation matrices (active description).
    Please notice that the Euler angles in the ang files follow passive
    description.

    Parameters:
    -----------
    R : numpy array shape(3, 3) or shape(N, 3, 3)
        Rotation matrix or list or rotation matrices

    conv : string (optional)
        Rotation convention
        Default: zxz (Bunge notation)

    **kwargs :
        verbose : boolean
            If True (default), print calculation time
        avg : boolean
            If True, calculates the Euler angles corresponding to the
            average orientation.
            If False (default), simply calculates the Euler angles for
            each rotation matrix provided.

    """

    Rdim = np.ndim(R)
    if Rdim == 2:
        R = R.reshape(1, 3, 3)

    if not kwargs.pop('avg', False):
        verbose = kwargs.pop('verbose', True)
        if verbose:
            t0 = time.time()
            sys.stdout.write('Calculating Euler angles... ')
            sys.stdout.flush()

        Phi = np.arccos(R[:, 2, 2])
        sPhi = np.sin(Phi)
        cphi1, cphi2 = -R[:, 1, 2]/sPhi, R[:, 2, 1]/sPhi
        sphi1, sphi2 = R[:, 0, 2]/sPhi, R[:, 2, 0]/sPhi

        # arctan2 returns value in the range [-pi,pi].
        phi1, phi2 = np.arctan2(sphi1, cphi1), np.arctan2(sphi2, cphi2)
        neg1, neg2 = phi1 < 0, phi2 < 0
        if np.ndim(neg1) > 0:
            # phi1 and phi2 to range [0, 2pi]
            phi1[neg1] = phi1[neg1] + 2.*np.pi
            phi2[neg2] = phi2[neg2] + 2.*np.pi
        else:
            if neg1:
                phi1 += 2.*np.pi
            if neg2:
                phi2 += 2.*np.pi

        if Rdim == 2:
            phi1, Phi, phi2 = phi1[0], Phi[0], phi2[0]

        if verbose:
            sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))
            sys.stdout.flush()
    else:
        Phi = np.arccos(np.mean(R[:, 2, 2]))
        sPhi = np.sin(Phi)
        cphi1, cphi2 = -np.mean(R[:, 1, 2])/sPhi, np.mean(R[:, 2, 1])/sPhi
        sphi1, sphi2 = np.mean(R[:, 0, 2])/sPhi, np.mean(R[:, 2, 0])/sPhi
        phi1, phi2 = np.arctan2(sphi1, cphi1), np.arctan2(sphi2, cphi2)
        R_avg = euler_angles_to_rotation_matrix(phi1, Phi, phi2, verbose=False)
        # n=kwargs.pop('n', 5), maxdev=kwargs.pop('maxdev', .25)
        R_avg = minimize_disorientation(R, R_avg, **kwargs)
        phi1, Phi, phi2 = rotation_matrix_to_euler_angles(R_avg, verbose=False)  # recursive

    return phi1, Phi, phi2


def axis_angle_to_rotation_matrix(axis, theta):
    theta_dim = np.ndim(theta)
    axis_dim = np.ndim(axis)

    if axis_dim != theta_dim + 1:
        raise Exception('Invalid shapes of theta or axis')

    if theta_dim == 0:
        theta = np.asarray(theta).reshape(-1)
        axis = np.asarray(axis).reshape(-1, 3)

    axis = axis/np.linalg.norm(axis, axis=1).reshape(-1, 1)

    N = len(theta)
    R = np.ndarray((N, 3, 3))

    ctheta = np.cos(theta)
    ctheta1 = 1 - ctheta
    stheta = np.sin(theta)

    R[:, 0, 0] = ctheta1*axis[:, 0]**2. + ctheta
    R[:, 0, 1] = ctheta1*axis[:, 0]*axis[:, 1] - axis[:, 2]*stheta
    R[:, 0, 2] = ctheta1*axis[:, 0]*axis[:, 2] + axis[:, 1]*stheta
    R[:, 1, 0] = ctheta1*axis[:, 1]*axis[:, 0] + axis[:, 2]*stheta
    R[:, 1, 1] = ctheta1*axis[:, 1]**2. + ctheta
    R[:, 1, 2] = ctheta1*axis[:, 1]*axis[:, 2] - axis[:, 0]*stheta
    R[:, 2, 0] = ctheta1*axis[:, 2]*axis[:, 0] - axis[:, 1]*stheta
    R[:, 2, 1] = ctheta1*axis[:, 2]*axis[:, 1] + axis[:, 0]*stheta
    R[:, 2, 2] = ctheta1*axis[:, 2]**2. + ctheta

    if theta_dim == 0:
        R = R.reshape(3, 3)

    return R


""" Symmetry operations for the cubic system """


def list_cubic_symmetry_operators_KS():
    """
    Lists symmetry matrices for cubic symmetry group following KS variants
    convention as represented in, e.g., 
    T. Furuhara, H. Kawata, S. Morito, T. Maki, Mater. Sci. Eng. A 431 (2006)
    228-236.
    """
    return np.array([[[1.,  0.,  0.],
                      [0.,  1.,  0.],
                      [0.,  0.,  1.]],

                     [[0.,  0., -1.],
                      [0., -1.,  0.],
                      [-1.,  0.,  0.]],

                     [[0.,  1.,  0.],
                      [0.,  0.,  1.],
                      [1.,  0.,  0.]],

                     [[0., -1.,  0.],
                      [-1.,  0.,  0.],
                      [0.,  0., -1.]],

                     [[0.,  0.,  1.],
                      [1.,  0.,  0.],
                      [0.,  1.,  0.]],

                     [[-1.,  0.,  0.],
                      [0.,  0., -1.],
                      [0., -1.,  0.]],

                     [[0.,  0.,  1.],
                      [0., -1.,  0.],
                      [1.,  0.,  0.]],

                     [[-1.,  0.,  0.],
                      [0.,  1.,  0.],
                      [0.,  0., -1.]],

                     [[1.,  0.,  0.],
                      [0.,  0., -1.],
                      [0.,  1.,  0.]],

                     [[0.,  0., -1.],
                      [1.,  0.,  0.],
                      [0., -1.,  0.]],

                     [[0.,  1.,  0.],
                      [-1.,  0.,  0.],
                      [0.,  0.,  1.]],

                     [[0., -1.,  0.],
                      [0.,  0.,  1.],
                      [-1.,  0.,  0.]],

                     [[0., -1.,  0.],
                      [1.,  0.,  0.],
                      [0.,  0.,  1.]],

                     [[0.,  1.,  0.],
                      [0.,  0., -1.],
                      [-1.,  0.,  0.]],

                     [[0.,  0., -1.],
                      [0.,  1.,  0.],
                      [1.,  0.,  0.]],

                     [[1.,  0.,  0.],
                      [0., -1.,  0.],
                      [0.,  0., -1.]],

                     [[-1.,  0.,  0.],
                      [0.,  0.,  1.],
                      [0.,  1.,  0.]],

                     [[0.,  0.,  1.],
                      [-1.,  0.,  0.],
                      [0., -1.,  0.]],

                     [[1.,  0.,  0.],
                      [0.,  0.,  1.],
                      [0., -1.,  0.]],

                     [[0.,  0., -1.],
                      [-1.,  0.,  0.],
                      [0.,  1.,  0.]],

                     [[0.,  1.,  0.],
                      [1.,  0.,  0.],
                      [0.,  0., -1.]],

                     [[0., -1.,  0.],
                      [0.,  0., -1.],
                      [1.,  0.,  0.]],

                     [[0.,  0.,  1.],
                      [0.,  1.,  0.],
                      [-1.,  0.,  0.]],

                     [[-1.,  0.,  0.],
                      [0., -1.,  0.],
                      [0.,  0.,  1.]]])


def list_cubic_symmetry_operators():
    """
    Lists symmetry matrices for cubic symmetry group
    """
    axis = np.array([[1., 0., 0.],
                     # 2-fold on <100>
                     [1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.],
                     # 4-fold on <100>
                     [1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.],
                     [-1., 0., 0.],
                     [0., -1., 0.],
                     [0., 0., -1.],
                     # 2-fold on <110>
                     [1., 1., 0.],
                     [1., 0., 1.],
                     [0., 1., 1.],
                     [1., -1., 0.],
                     [-1., 0., 1.],
                     [0., 1., -1.],
                     # 3-fold on <111>
                     [1., 1., 1.],
                     [1., -1., 1.],
                     [-1., 1., 1.],
                     [-1., -1., 1.],
                     [1., 1., -1.],
                     [1., -1., -1.],
                     [-1., 1., -1.],
                     [-1., -1., -1.]])

    angle = np.array([0.,
                      # 2-fold on <100>
                      np.pi,
                      np.pi,
                      np.pi,
                      # 4-fold on <100>
                      np.pi/2.,
                      np.pi/2.,
                      np.pi/2.,
                      np.pi/2.,
                      np.pi/2.,
                      np.pi/2.,
                      # 2-fold on <110>
                      np.pi,
                      np.pi,
                      np.pi,
                      np.pi,
                      np.pi,
                      np.pi,
                      # 3-fold on <111>
                      np.pi*2./3.,
                      np.pi*2./3.,
                      np.pi*2./3.,
                      np.pi*2./3.,
                      np.pi*2./3.,
                      np.pi*2./3.,
                      np.pi*2./3.,
                      np.pi*2./3.])

    # Round and convert float to int. The elements of the operators are
    # the integers 0, 1, and -1
    return axis_angle_to_rotation_matrix(axis, angle).round(0).astype(int)


def list_cubic_family_directions(d):
    """
    Lists all the variants of a family of directions 'd'.
    """
    C = list_cubic_symmetry_operators()
    var = set([tuple(v) for v in np.dot(C, d)])
    return np.asarray(list(var))


def reduce_cubic_transformations(V, maxdev=1e-3):
    """
    Remove redudant transformations (rotations) and returns a reduced
    number of matrices.

    Parameters
    ----------
    V : ndarray shape(N, 3, 3)
        List of N 3x3 arrays (matrices) representing crystal bases.
   
    Returns
    -------
    Vprime : ndarray shape(K, 3, 3), K <= N
        Subset of V containing only the non-reduntant rotations
    """
    # Convert maxdev from angle in degrees to trace values [-1, 3]. Because
    # maxdev is very small, the new value of maxdev is very close to 3.
    maxdev = 2.*np.cos(np.radians(maxdev)) + 1.
    C = list_cubic_symmetry_operators()

    pairs = []
    N = len(V)
    for i in range(N):
        p = []
        U = np.dot(C, V[i])
        for j in range(i, N):
            tr = np.abs(np.trace(np.dot(U, V[j].T), axis1=1, axis2=2))
            # From the trace tr you can get the misorientation angle
            # The following "if" is equivalent to check if the misorientation
            # angle is less the angle "maxdev"
            if tr.max() >= maxdev:
                p.append(j)
        pairs.append(p)

    unique = [p.pop() for p in pairs if len(p) == 1]
    Vprime = V[unique]

    return Vprime


""" Orientation representation (IPF and PF) """


def IPF(M, d=[0, 0, 1]):
    """
    Calculates crystallographic direction parallel to the direction d 
    relative to the sample coordinate frame.

    Parameters
    ----------
    M : numpy ndarray shape(N,3,3)
        Rotation matrices describing the transformation from the sample 
        coordinate frame to the crystal coordinate frame
    d : list or array shape(3)
        Reference direction in the sample coordinate frame.

    Returns
    -------
    uvw : crystallographic direction parallel to the direction 'd'
        uvw = M.d = (R^T).d
    """
    if np.ndim(M) == 2:
        M = M.reshape(1, 3, 3)

    uvw = np.dot(M, d)  # dot product M.D

    return uvw/np.linalg.norm(uvw, axis=1).reshape(-1, 1)  # normalize uvw


def PF(R, proj=[1, 0, 0], rotation=None):
    """
    Parameters
    ----------
    R : numpy ndarray shape(N,3,3) or numpy array(3,3)
        Rotation matrices describing the transformation from the crystal 
        coordinate frame to the sample coordinate frame
    proj : list or array shape(3)
        Family of directions projected in the pole figure. Default is '100'
    rotation : list or array shape(3,3)
        Rotation matrix that rotates the pole figure (R' = rotation-1.R).
        The columns of the matrix correspond to the directions parallel to 
        the axes of the pole figure.
    """
    if np.ndim(R) == 2:
        R = R.reshape(1, 3, 3)

    if isinstance(rotation, (list, tuple, np.ndarray)):
        R_prime = rotation/np.linalg.norm(rotation, axis=0)
        R_prime = np.linalg.inv(R_prime)
        R = np.tensordot(R_prime, R, axes=[[-1], [-2]]).transpose([1, 0, 2])

    N = R.shape[0]
    proj_variants = list_cubic_family_directions(d=proj)
    nvar = len(proj_variants)
    # normalize proj_variants
    proj_variants = proj_variants/np.linalg.norm(proj)

    # Return directions in the sample coordinate frame ndarray shape(N, nvar, 3)
    return np.tensordot(R, proj_variants.T, axes=[[-1], [-2]]).transpose([0, 2, 1])

    # xp = np.ndarray((N, nvar))
    # yp = np.ndarray((N, nvar))

    # dm : directions in the sample coordinate frame
    # ndarray shape(N,3,nvar)
    # dm = np.tensordot(R, proj_variants.T, axes=[[-1], [-2]])
    # sgn = np.sign(dm[:, 2, :])  # ndarray shape(N, nvar)
    # sgn[sgn == 0.] = 1.  # change behavior of np.sign for x = 0

    # # coordinates in the stereographic projection
    # xp = sgn*dm[:, 0, :]/(np.abs(dm[:, 2, :]) + 1.)
    # yp = sgn*dm[:, 1, :]/(np.abs(dm[:, 2, :]) + 1.)

    # return (xp, yp)


# The product of two 2D matrices (numpy ndarray shape(N,N)) can be calculated
# using the function 'numpy.dot'. In order to compute the matrix product of
# higher dimensions arrays, numpy.dot can also be used, but paying careful
# attention to the indices of the resulting matrix. Examples:
#     - A is ndarray shape(N,M,3,3) and B is ndarray shape(3,3):
#     np.dot(A,B)[i,j,k,m] = np.sum(A[i,j,:,k]*B[m,:])
#     np.dot(A,B) is ndarray shape(N,M,3,3)

#     - A is ndarray shape(N,3,3) and B is ndarray shape(M,3,3):
#     np.dot(A,B)[i,j,k,m] = np.sum(A[i,:,j]*B[k,m,:])
#
#     The result np.dot(A,B) is ndarray shape(N,3,M,3). It's more convenient to
#     express the result as ndarray shape(N,M,3,3). In order to obtain the
#     desired result, the 'transpose' function should be used. i.e.,
#     np.dot(A,B).transpose([0,2,1,3]) results in ndarray shape(N,M,3,3)

#     - A is ndarray shape(3,3) and B is ndarray shape(N,M,3,3):
#     np.dot(A,B)[i,j,k,m] = np.sum(A[:,i]*B[j,k,m,:])
#
#     np.dot(A,B) is ndarray shape(3,N,M,3). np.dot(A,B).transpose([1,2,0,3])
#     results in ndarray shape(N,M,3,3)

# 'numpy.dot' is a particular case of 'numpy.tensordot':
# np.dot(A,B) == np.tensordot(A, B, axes=[[-1],[-2]])

# numpy.tensordot is two times faster than numpy.dot

# http://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
