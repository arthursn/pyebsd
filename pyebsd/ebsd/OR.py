import sys
import time
import numpy as np

from .orientation import (list_cubic_symmetry_operators,
                          reduce_cubic_transformations, average_orientation,
                          rotation_matrix_to_euler_angles,
                          euler_angles_to_rotation_matrix)


__all__ = ['OR_exp', 'OR']


def OR_exp(M, ph, phdict=dict(parent=2, child=1), sel=None, **kwargs):
    """
    Calculates the accurate orientation relationship between parent
    and child phases.

    - Miyamoto, G., Takayama, N., & Furuhara, T. (2009). Accurate
      measurement of the orientation relationship of lath martensite
      and bainite by electron backscatter diffraction analysis.
      Scripta Materialia, 60(12), 1113-1116.
      http://doi.org/10.1016/j.scriptamat.2009.02.053

    Parameters
    ----------
    M : numpy ndarray shape(N, 3, 3)
        List of rotation matrices describing the rotation from the sample
        coordinate frame to the crystal coordinate frame
    ph : numpy ndarray shape(N)
        Array containing the phase indexes corresponding to the
        rotation matrices provided in M
    phdict : dict
        Dictionary mapping the phase indexes to the parent and child
        (product) phases, e.g.: phdict=dict(parent=2, child=1),
        meaning thatn in ph the parent phase is indexed as 2 and the
        product phase as 1.
    sel : bool numpy 1D array (optional)
        Boolean array indicating data points calculations should be
        performed
        Default: None
    **kwargs :
        verbose : bool (optional)
            If True, prints computation time
            Default: True
        C : numpy ndarray shape(24, 3, 3) (optional)
            List of 24 cubic symmetry operators. This option might be
            useful because the OR matrices calculated using different
            symmetry operators (Vj = V0 * C^-1) lead to crystallographic
            variants in different conventions.
            For instance, the symmetry operators calculated using
            pyebsd.list_cubic_symmetry_operators_KS() result in the
            variants sorted according to the convention of papers such as
            T. Furuhara, H. Kawata, S. Morito, T. Maki, Mater. Sci. Eng. A
            431 (2006) 228-236.
            If none is provided, C is calculated with
            pyebsd.list_cubic_symmetry_operators()

    Returns
    -------
    V, Vavg, Mprt, variants : tuple
        V : numpy ndarray shape(N, 3, 3)
            List of orientation relationship matrices for each pixel
        Vavg : numpy ndarray shape(3, 3)
            The average orientation relationship of V
        Mprt : numpy ndarray shape(3, 3)
            The average orientation of the parent phase
        variants : numpy ndarray shape(N)
            The variants corresponding to each pixel

    """
    t0 = time.time()
    verbose = kwargs.pop('verbose', True)
    if verbose:
        sys.stdout.write('Calculating variants... ')
        sys.stdout.flush()

    if isinstance(phdict, dict):
        prt, chd = phdict['parent'], phdict['child']
    else:
        prt, chd = phdict[0], phdict[1]

    if not isinstance(sel, np.ndarray):
        sel = np.full(len(M), True)

    # Calculate average rotation matrix of parent phase
    Mprt = average_orientation(M, sel=sel & (ph == prt), verbose=False)
    # Rotation matrices of child phases
    Mchd = M[sel & (ph == chd)]

    N = len(Mchd)

    # Get symmetry operators
    C = kwargs.pop('C', list_cubic_symmetry_operators())
    # T = C * Mprt : ndarray shape(24, 3, 3)
    T = np.tensordot(C, Mprt, axes=[[-1], [-2]]).transpose([0, 2, 1])
    # U = Mchd * T^-1 : ndarray shape(N, 24, 3, 3)
    U = np.tensordot(Mchd, T, axes=[[-1], [-2]]).transpose([0, 2, 1, 3])

    Nsym = len(C)
    variants = np.ndarray((Nsym, N), dtype=int)
    Vsel = np.ndarray((Nsym, N, 3, 3))
    trmax = np.ndarray((Nsym, N))

    VrefT = np.tensordot(C[0], U[0, 0], axes=1).T  # axes=[[-1],[-2]] also works
    # This step is non vectorized to save memory
    for i in range(len(C)):
        V = np.tensordot(C[i], U, axes=[[-1], [-2]]).transpose([1, 2, 0, 3])
        D = np.tensordot(V, VrefT, axes=[[-1], [-2]])
        tr = np.trace(D, axis1=2, axis2=3)
        # neg = tr < 0.
        # tr[neg] = -tr[neg]
        # V[neg] = -V[neg]

        variants[i] = np.argmax(tr, axis=1)
        trmax[i] = np.max(tr, axis=1)
        Vsel[i] = V[(list(range(N)), variants[i])]

    jsel = (np.argmax(trmax, axis=0), list(range(N)))
    variants = variants[jsel]
    tr = trmax[jsel]
    V = Vsel[jsel]

    # Euler angles of average orientation
    phi1, Phi, phi2 = rotation_matrix_to_euler_angles(Vsel[jsel], avg=True, verbose=False)
    # Average orientation rotation matrix
    Vavg = euler_angles_to_rotation_matrix(phi1, Phi, phi2, verbose=False)

    if verbose:
        sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))

    # Delete arrays
    del Mchd, T, U, D, Vsel

    # Return the OR matrices V for each pixel,
    # the average OR matrix Vavg, and the
    # rotation matrix of the parent phase
    return V, Vavg, Mprt, variants


def OR(ps=([1, 1, 1], [0, 1, 1]), ds=([-1, 0, 1], [-1, -1, 1]), single=False, **kwargs):
    """
    From the parallel planes (ps) and directions (ds) determine the
    orientations matrices of the parent (Mprt) and child (Mchd)
    phases. Having Mprt and Mchd, calculates all the transformation
    matrices V (all the variants) of the orientation relationship
    between the two phases.
    By default, generates the 24 Kurjumov-Sachs orientation matrices

    Parameters
    ----------
    ps : list or tuple containing 2 lists shape(3) (optional)
        Miller indices of the parallel planes. The first on the list
        corresponds to the parent phase, while the second correspond
        to the plane in the product phase
        Default : ([1, 1, 1], [0, 1, 1]) (close-packed planes)
    ds : list or tuple containing 2 lists shape(3) (optional)
        Miller indices of parallel directions. The first on the list
        corresponds to the parent phase, while the second correspond
        to the plane in the product phase
        Default : ([-1, 0, 1], [-1, -1, 1]) (close-packed directions)
    single : bool (optional)
        If True, returns only the orientation matrix of the first
        variant
    **kwargs :
        C : numpy ndarray shape(24, 3, 3) (optional)
            List of 24 cubic symmetry operators. This option might be
            useful because the OR matrices calculated using different
            symmetry operators (Vj = V0 * C^-1) lead to crystallographic
            variants in different conventions.
            For instance, the symmetry operators calculated using
            pyebsd.list_cubic_symmetry_operators_KS() result in the
            variants sorted according to the convention of papers such as
            T. Furuhara, H. Kawata, S. Morito, T. Maki, Mater. Sci. Eng. A
            431 (2006) 228-236.
            If none is provided, C is calculated with
            pyebsd.list_cubic_symmetry_operators()

    Returns
    -------
    V : numpy ndarray
        If single is False, returns the orientation matrices of all
        variants of the orientation relationship. If single is True,
        then returns only the orientation matrix of the first
        variant

    """
    trunc = kwargs.pop('trunc', 1e-8)
    ps, ds = np.asarray(ps), np.asarray(ds)
    p_prt, d_prt = ps[0], ds[0]  # parent phase
    p_chd, d_chd = ps[1], ds[1]  # child phase

    C = kwargs.pop('C', list_cubic_symmetry_operators())

    # check variants normal to plane 'n'. Due to numerical truncation,
    # instead of choosing the variants 'd' based on np.dot(d,n) == 0,
    # a tolerance 'trunc' is set. i.e., the variants are chosen
    # according to the criteria np.abs(np.dot(d,n)) <= trunc (1e-8)
    ds = np.dot(C, d_prt)
    sel = np.abs(np.asarray([np.dot(p_prt, d) for d in ds])) <= trunc
    d_prt = ds[sel][0]

    ds = np.dot(C, d_chd)
    sel = np.abs(np.asarray([np.dot(p_chd, d) for d in ds])) <= trunc
    d_chd = ds[sel][0]

    R_prt = np.array([d_prt, -np.cross(d_prt, p_prt), p_prt])
    R_chd = np.array([d_chd, -np.cross(d_chd, p_chd), p_chd])

    R_prt = R_prt/np.linalg.norm(R_prt, axis=1).reshape(-1, 1)
    R_chd = R_chd/np.linalg.norm(R_chd, axis=1).reshape(-1, 1)

    V = np.dot(R_chd.T, R_prt)

    if not single:
        V = np.matmul(V, C.transpose(0, 2, 1))
        V = reduce_cubic_transformations(V)

    return V
