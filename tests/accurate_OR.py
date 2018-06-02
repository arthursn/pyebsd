from pyebsd import list_symm, list_vars, OR, OR_exp, tr2ang
from itertools import combinations

def angle_2ds(V, ds='cpp'):
    if type(ds) == str:
        ds = ds.lower()
        if ds == 'cpp':
            d_prt, d_chd = [1,1,1], [1,1,0]
        elif ds == 'cpd':
            d_prt, d_chd = [1,1,0], [1,1,1]
        else:
            print('Non recognized option')
            return
    else:
        d_prt, d_chd = ds[0], ds[1]

    v = np.dot(V, list_vars(d_prt).T)/np.linalg.norm(d_prt) # directions
    v = np.dot(list_vars(d_chd), v)/np.linalg.norm(d_chd) # cosine 
    v = np.abs(v)
    v[v > 1] = 1. # if v[i] == 1.000...1 np.arccos crashes
    v = np.degrees(np.arccos(v)) # angle in degrees
    if np.ndim(v) == 2:
        v = np.min(v) # minimum angle
    elif np.ndim(v) == 3:
        v = np.min(np.min(v, axis=0), axis=1)
    else:
        return
    return v

filt = sel & (scan.CI > .2)
V, V0, R_fcc = OR_exp(R=scan.R, ph=scan.ph, sel=filt)

C = list_symm()
KS = OR()
Vr = KS[1] # reference

# D = np.dot(np.dot(C, V0), np.dot(C, Vr).transpose([0,2,1])).transpose([0,2,1,3])
A = np.dot(np.dot(C, V0), C).transpose([0,2,1,3])
D = np.dot(A, Vr.T)
tr = np.trace(D, axis1=2, axis2=3)
nm = np.argmax(np.abs(tr))
n, m = np.unravel_index(nm, tr.shape)

V0, V = np.dot(np.dot(C[n], V0), C[m]), np.dot(np.dot(C[n], V).transpose([1,0,2]), C[m])
if tr[n, m] < 0:
    V0, V = -V0, -V


D_KS = np.dot(V, Vr.T) # deviation from KS
tr_KS = np.trace(D_KS, axis1=1, axis2=2)
mis_KS = np.ndarray(scan.N)
mis_KS.fill(0)
mis_KS[filt & (scan.ph==1)] = tr2ang(tr_KS)
mapks = scan.plot_property(mis_KS, sel=filt & (scan.ph==1), vmin=0, vmax=5., gray=scan.IQ, tiling='hex')

# D0 = np.dot(V, V0.T) # deviation from average
# tr0 = np.trace(D0, axis1=1, axis2=2)
# mis0 = np.ndarray(scan.N)
# mis0.fill(0)
# mis0[filt & (scan.ph==1)] = tr2ang(tr0)
# scan.plot_property(mis0, sel=filt & (scan.ph==1), vmin=0, vmax=3, gray=scan.IQ, tiling='hex')

# cpp = np.ndarray(scan.N)
# cpd = np.ndarray(scan.N)
# cpp.fill(0)
# cpd.fill(0)

# cpp[filt & (scan.ph==1)] = angle_2ds(V, 'cpp')
# cpd[filt & (scan.ph==1)] = angle_2ds(V, 'cpd')

# scan.plot_property(cpp, sel=filt & (scan.ph==1), vmin=0, vmax=3, gray=scan.IQ, tiling='hex')
# scan.plot_property(cpd, sel=filt & (scan.ph==1), vmin=2, vmax=5, gray=scan.IQ, tiling='hex')


print('Current')
print('cpp: {:.4f} deg'.format(angle_2ds(V0, 'cpp')))
print('cpd: {:.4f} deg'.format(angle_2ds(V0, 'cpd')))

T = OR()[0]
print('KS')
print('cpp: {:.4f} deg'.format(angle_2ds(T, 'cpp')))
print('cpd: {:.4f} deg'.format(angle_2ds(T, 'cpd')))

T = OR(ds=([1,1,0],[1,0,0]))[0]
print('NW')
print('cpp: {:.4f} deg'.format(angle_2ds(T, 'cpp')))
print('cpd: {:.4f} deg'.format(angle_2ds(T, 'cpd')))

T = OR(ds=([5,12,17],[7,17,17]))[0]
print('GT')
print('cpp: {:.4f} deg'.format(angle_2ds(T, 'cpp')))
print('cpd: {:.4f} deg'.format(angle_2ds(T, 'cpd')))

figx, axx = plt.subplots()
axx.hist(angle_2ds(V, 'cpp'), bins=150)
axx.hist(-angle_2ds(V, 'cpd'), bins=150)
axx.set_xlim(-8,8)
axx.set_xlabel('Deviation from closed packed plane (+) or direction (-) [deg]')
axx.set_ylabel('Counts')

proj0, proj = [1,1,0], [1,1,1]

ax = pyebsd.plot_PF(R=scan.R[filt & (scan.ph == 2)], contour=True, cmap=plt.get_cmap('Blues_r'), bins=512, nlevels=20, fn=None, proj=proj0, parent_or=R_fcc)
pyebsd.plot_PF(M=V, contour=True, cmap=plt.get_cmap('Reds_r'), bins=512, nlevels=20, fn=None, proj=proj, ax=ax)
# ax = pyebsd.plot_PF(R=scan.R[filt & (scan.ph == 2)], c="b", proj=proj0, parent_or=R_fcc)
# pyebsd.plot_PF(M=V, c="r", proj=proj, ax=ax)
pyebsd.plot_PF(M=np.dot(V0, C).transpose([1,0,2]), mfc=[1,1,1,0], mec='k', mew=1, marker='o', ms=3, proj=proj, verbose=False, ax=ax)
pyebsd.draw_std_traces(ax, lw=.2)
pyebsd.draw_wulff_net(ax, lw=.2)

#####################

parent_or = np.diag([1,1,1])
axis0, axis1 = np.asarray(parent_or)[:,0], np.asarray(parent_or)[:,1]
# ax.annotate(uvw_label(axis0, s='\gamma'), xy=(1,0), ha='left', va='center')
# ax.annotate(uvw_label(axis1, s='\gamma'), xy=(0,1), ha='center', va='bottom')
# fig4 = plt.gcf()

T_KS = pyebsd.OR() # K-S
T_NW = pyebsd.OR(ds=[[1,1,0],[1,0,0]]) # N-W
T_GT = pyebsd.OR(ds=[[5,12,17],[7,17,17]]) # G-T

proj = [1,0,0]

ax2 = pyebsd.plot_PF(M=np.dot(V0, C).transpose([1,0,2]), mfc=[1,1,1,0], mec='k', mew=1, marker='o', ms=5, proj=proj, label='exp.')
pyebsd.plot_PF(M=T_KS, mfc=[1,1,1,0], mec='r', mew=1, marker='s', ms=5, proj=proj, parent_or=parent_or, verbose=False, ax=ax2, label='KS')
pyebsd.plot_PF(M=T_NW, mfc=[1,1,1,0], mec='g', mew=1, marker='D', ms=5, proj=proj, parent_or=parent_or, verbose=False, ax=ax2, label='NW')
pyebsd.plot_PF(M=T_GT, mfc=[1,1,1,0], mec='b', mew=1, marker='v', ms=5, proj=proj, parent_or=parent_or, verbose=False, ax=ax2, label='GT')
pyebsd.draw_std_traces(ax2, lw=.2)
pyebsd.draw_wulff_net(ax2, lw=.2)
ax2.legend()

# ax2.annotate(uvw_label(axis0, s='\gamma'), xy=(1,0), ha='left', va='center')
# ax2.annotate(uvw_label(axis1, s='\gamma'), xy=(0,1), ha='center', va='bottom')
# fig5 = plt.gcf()


plt.show()

