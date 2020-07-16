from pyebsd import list_cubic_symmetry_operators_KS, list_cubic_family_directions, OR, OR_exp, trace_to_angle
from itertools import combinations


def angle_2ds(V, ds='cpp'):
    if type(ds) == str:
        ds = ds.lower()
        if ds == 'cpp':
            d_prt, d_chd = [1, 1, 1], [1, 1, 0]
        elif ds == 'cpd':
            d_prt, d_chd = [1, 1, 0], [1, 1, 1]
        else:
            print('Non recognized option')
            return
    else:
        d_prt, d_chd = ds[0], ds[1]

    v = np.dot(V, list_cubic_family_directions(d_prt).T) / \
        np.linalg.norm(d_prt)  # directions
    v = np.dot(list_cubic_family_directions(d_chd), v) / \
        np.linalg.norm(d_chd)  # cosine
    v = np.abs(v)
    v[v > 1] = 1.  # if v[i] == 1.000...1 np.arccos crashes
    v = np.degrees(np.arccos(v))  # angle in degrees
    if np.ndim(v) == 2:
        v = np.min(v)  # minimum angle
    elif np.ndim(v) == 3:
        v = np.min(np.min(v, axis=0), axis=1)
    else:
        return
    return v


# Calculate experimental (accurate?) OR
filt = sel & (scan.CI > .2)
V, V0, M_fcc, jvariants = OR_exp(M=scan.M, ph=scan.ph, sel=filt)

C = list_cubic_symmetry_operators_KS()
KS = OR()
Vr = KS[2]  # reference

# D = np.dot(np.dot(C, V0), np.dot(C, Vr).transpose([0,2,1])).transpose([0,2,1,3])
A = np.dot(np.dot(C, V0), C).transpose([0, 2, 1, 3])
D = np.dot(A, Vr.T)
tr = np.trace(D, axis1=2, axis2=3)
nm = np.argmax(np.abs(tr))
n, m = np.unravel_index(nm, tr.shape)

V0, V = np.dot(np.dot(C[n], V0), C[m]), np.dot(
    np.dot(C[n], V).transpose([1, 0, 2]), C[m])
if tr[n, m] < 0:
    V0, V = -V0, -V


# Calculate and plot map of the deviation from KS OR
D_KS = np.dot(V, Vr.T)  # deviation from KS
tr_KS = np.trace(D_KS, axis1=1, axis2=2)
mis_KS = np.zeros(scan.N)
mis_KS[filt & (scan.ph == 1)] = trace_to_angle(tr_KS)
mapks = scan.plot_property(mis_KS, sel=filt & (scan.ph == 1),
                           vmin=0, vmax=5., gray=scan.IQ, tiling='hex')
ax7, fig7 = mapks.ax, mapks.fig
# fig7.savefig(basename + '_mis_KS.pdf', bbox_inches='tight', pad_inches=0)

# Plot crystallographic variants
variants = np.zeros(scan.N)
variants[filt & (scan.ph == 1)] = jvariants
mapvars = scan.plot_property(variants, sel=filt & (scan.ph == 1),
                             vmin=0, vmax=23, gray=scan.IQ, tiling='hex',
                             cmap=plt.get_cmap('jet'))

ax9, fig9 = mapvars.ax, mapvars.fig
# fig9.savefig(basename + '_variants.pdf', bbox_inches='tight', pad_inches=0)

# Plot histogram of deviation from KS
fig8, ax8 = plt.subplots()
ax8.hist(mis_KS[filt & (scan.ph == 1)], bins=150)

ax8.set_xlim(0, 10)
ylim = ax8.get_ylim()

mis_avg = np.mean(mis_KS[filt & (scan.ph == 1)])
ax8.axvline(mis_avg, ls='--', color='k')
ax8.annotate(s=u'Mean: {:.2g}°'.format(mis_avg),
             xy=(mis_avg + .5, ylim[1]*.9),
             ha='left')

ax8.set_xlabel(u'Deviation from KS OR (°)')
ax8.set_ylabel('Counts')
# fig8.savefig(basename + '_mis_KS_hist.pdf')

# np.savetxt(basename + '_mis_KS.txt', mis_KS[filt & (scan.ph == 1)], fmt='%.6e')


print('Experimental')
print('cpp: {:.4f} deg'.format(angle_2ds(V0, 'cpp')))
print('cpd: {:.4f} deg'.format(angle_2ds(V0, 'cpd')))

T = OR()[0]
print('KS')
print('cpp: {:.4f} deg'.format(angle_2ds(T, 'cpp')))
print('cpd: {:.4f} deg'.format(angle_2ds(T, 'cpd')))

T = OR(ds=([1, 1, 0], [1, 0, 0]))[0]
print('NW')
print('cpp: {:.4f} deg'.format(angle_2ds(T, 'cpp')))
print('cpd: {:.4f} deg'.format(angle_2ds(T, 'cpd')))

T = OR(ds=([5, 12, 17], [7, 17, 17]))[0]
print('GT')
print('cpp: {:.4f} deg'.format(angle_2ds(T, 'cpp')))
print('cpd: {:.4f} deg'.format(angle_2ds(T, 'cpd')))

fig4, ax4 = plt.subplots()
ax4.hist(angle_2ds(V, 'cpp'), bins=150)
ax4.hist(-angle_2ds(V, 'cpd'), bins=150)
ax4.set_xlim(-8, 8)
ax4.set_xlabel(('Deviation from closed packed direction (-) '
                'or closed packed plane (+) (deg)'))
ax4.set_ylabel('Counts')

proj0, proj = [1, 1, 0], [1, 1, 1]

ax5 = scan.plot_PF(proj=proj0, sel=filt & (scan.ph == 2), rotation=M_fcc.T,
                   contour=True, cmap='Blues_r', bins=512, nlevels=20)
fig5 = ax5.get_figure()
pyebsd.plot_PF(M=V, contour=True, cmap='Reds_r',
               bins=512, nlevels=20, fn=None, proj=proj, ax=ax5)
pyebsd.plot_PF(M=np.dot(V0, C).transpose([1, 0, 2]),
               mfc=[1, 1, 1, 0], mec='k', mew=1, marker='o',
               ms=3, proj=proj, verbose=False, ax=ax5)
pyebsd.draw_std_traces(ax5, lw=.2)
pyebsd.draw_wulff_net(ax5, lw=.2)

#####################

parent_or = np.diag([1, 1, 1])
axis0, axis1 = np.asarray(parent_or)[:, 0], np.asarray(parent_or)[:, 1]

T_KS = pyebsd.OR()  # K-S
T_NW = pyebsd.OR(ds=[[1, 1, 0], [1, 0, 0]])  # N-W
T_GT = pyebsd.OR(ds=[[5, 12, 17], [7, 17, 17]])  # G-T

proj = [1, 0, 0]

ax6 = pyebsd.plot_PF(M=np.dot(V0, C).transpose([1, 0, 2]),
                     mfc=[1, 1, 1, 0], mec='k', mew=1, marker='o',
                     ms=5, proj=proj, label='exp.')
fig6 = ax6.get_figure()
pyebsd.plot_PF(M=T_KS, mfc=[1, 1, 1, 0], mec='r', mew=1,
               marker='s', ms=5, proj=proj, rotation=parent_or,
               verbose=False, ax=ax6, label='KS')
pyebsd.plot_PF(M=T_NW, mfc=[1, 1, 1, 0], mec='g', mew=1,
               marker='D', ms=5, proj=proj, rotation=parent_or,
               verbose=False, ax=ax6, label='NW')
pyebsd.plot_PF(M=T_GT, mfc=[1, 1, 1, 0], mec='b', mew=1,
               marker='v', ms=5, proj=proj, rotation=parent_or,
               verbose=False, ax=ax6, label='GT')
pyebsd.draw_std_traces(ax6, lw=.2)
pyebsd.draw_wulff_net(ax6, lw=.2)
ax6.legend()


plt.show()
