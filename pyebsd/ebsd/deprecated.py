# def OR_exp(R, ph, phdict=dict(parent=2, child=1), sel=None, **kwargs):
#     """
#     Experimental orientation relationship
#     """
#     t0 = time.time()
#     verbose = kwargs.pop('verbose', True)
#     if verbose:
#         sys.stdout.write('Calculating variants... ')
#         sys.stdout.flush()

#     if isinstance(phdict, dict):
#         prt, chd = phdict['parent'], phdict['child']
#     else:
#         prt, chd = phdict[0], phdict[1]

#     if not isinstance(sel, np.ndarray):
#         sel = np.ndarray(len(R), dtype=bool)
#         sel[:] = True

#     R_prt = avg_orientation(R, sel=sel & (ph == prt), verbose=False)
#     R_chd = R[sel & (ph == chd)]

#     M_prt = R_prt.T
#     M_chd = R_chd.transpose([0,2,1])
#     N = len(M_chd)

#     C = list_cubic_symmetry_operators()
#     T = np.tensordot(C, M_prt, axes=[[-1],[-2]]).transpose([0,2,1]) # T : ndarray shape(24, 3, 3)
#     U = np.tensordot(M_chd, T, axes=[[-1],[-2]]).transpose([0,2,1,3]) # U : ndarray shape(N, 24, 3, 3)

#     isel = np.ndarray((24,N), dtype=int)
#     V_sel = np.ndarray((24,N,3,3))
#     trmax = np.ndarray((24,N))

#     VrefT = np.tensordot(C[0], U[0,0], axes=1).T # axes=[[-1],[-2]] also works
#     # This step is non vectorized to save memory
#     for i in range(len(C)):
#         V = np.tensordot(C[i], U, axes=[[-1],[-2]]).transpose([1,2,0,3])
#         D = np.tensordot(V, VrefT, axes=[[-1],[-2]])
#         tr = np.trace(D, axis1=2, axis2=3)
#         neg = tr < 0.
#         tr[neg] = -tr[neg]
#         V[neg] = -V[neg]

#         isel[i] = np.argmax(tr, axis=1)
#         trmax[i] = np.max(tr, axis=1)
#         V_sel[i] = V[[list(range(N)), isel[i]]]

#     jsel = [np.argmax(trmax, axis=0), list(range(N))]
#     isel = isel[jsel]
#     tr = trmax[jsel]
#     V = V_sel[jsel]

#     phi1, Phi, phi2 = euler_angles(V_sel[jsel], avg=True, **kwargs)
#     V0 = euler_rotation(phi1, Phi, phi2, verbose=False)

#     if verbose:
#         sys.stdout.write('{:.2f} s\n'.format(time.time() - t0))

#     del R_chd, M_prt, M_chd, T, U, D, V_sel
#     return V, V0, R_prt

# def OR(ps=([1,1,1],[0,1,1]), ds=([0,1,1],[1,1,1]), trunc=1e-8):
#     """
#     From the parallel planes (ps) and directions (ds) determine the orientations
#     matrices of the parent (M_prt) and child (M_chd) phases. Having M_prt and M_chd,
#     calculates all the transformation matrices V (all the variants) of the
#     orientation relationship between the two phases.
#     """
#     ps, ds = np.asarray(ps), np.asarray(ds)
#     p_prt, d_prt = ps[0], ds[0] # parent phase
#     p_chd, d_chd = ps[1], ds[1] # child phase

#     # check variants normal to plane 'n'. Due to numerical truncation, instead
#     # of choosing the variants 'd' based on np.dot(d,n) == 0, a tolerance 'trunc'
#     # is set. i.e., the variants are chosen according to the criteria
#     # np.abs(np.dot(d,n)) <= trunc (1e-8)
#     if np.abs(np.dot(p_prt, d_prt)) > trunc:
#         ds = list_cubic_family_directions(d_prt)
#         sel = np.abs(np.asarray([np.dot(p_prt, d) for d in ds])) <= trunc
#         d_prt = ds[sel][0]
#     if np.abs(np.dot(p_chd, d_chd)) > trunc:
#         ds = list_cubic_family_directions(d_chd)
#         sel = np.abs(np.asarray([np.dot(p_chd, d) for d in ds])) <= trunc
#         d_chd = ds[sel][0]

#     p_prt = p_prt/np.linalg.norm(p_prt)
#     p_chd = p_chd/np.linalg.norm(p_chd)
#     d_prt = d_prt/np.linalg.norm(d_prt)
#     d_chd = d_chd/np.linalg.norm(d_chd)

#     M_prt = np.array([d_prt,-np.cross(d_prt,p_prt),p_prt]).T
#     M_chd = np.array([d_chd,-np.cross(d_chd,p_chd),p_chd]).T

#     V0 = np.dot(M_chd, M_prt.T)
#     C = list_cubic_symmetry_operators()
#     V = np.tensordot(V0, C, axes=[[-1],[-2]]).transpose([1,0,2])

#     return reduce_vars(V)
