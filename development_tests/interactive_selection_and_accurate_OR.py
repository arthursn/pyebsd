import os
import numpy as np
import matplotlib.pyplot as plt
import pyebsd


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

    v = np.dot(V, pyebsd.list_cubic_family_directions(d_prt).T) / \
        np.linalg.norm(d_prt)  # directions
    v = np.dot(pyebsd.list_cubic_family_directions(d_chd), v) / \
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


if __name__ == '__main__':
    # fname = os.path.join('..', 'data', 'ADI_bcc_fcc.ang')
    fname = os.path.join('..', 'data', 'QP_bcc_fcc_single_austenite_grain.ang')

    # basename = os.path.basename(fname)
    # basename = os.path.splitext(basename)[0]

    if 'scan' not in globals() or 'scan' not in locals():
        scan = pyebsd.load_scandata(fname)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    fig.canvas.manager.set_window_title(('Select area and press SHIFT for plotting selection. '
                                         'Press ENTER to continue with accurate OR calculation.'))

    # Plot unit triangle
    pyebsd.unit_triangle(ax3)
    ax3.set_xlim(-.1, 0.5)
    ax3.set_ylim(-.1, 0.4)

    ipf = scan.plot_IPF(sel=(scan.ph == 2) & (scan.CI > .2), gray=scan.IQ, d=[0, 0, 1], ax=ax1)

    # select points using lasso or rectangle selector. Interactive mode should be on
    lasso = ipf.lasso_selector()

    plt.ion()  # Interactive mode on
    plt.show()

    proj = [1, 0, 0]

    class Tracker:
        pass

    def calculate_accurate_OR(*args):
        if np.count_nonzero(ipf.sel) == 0:
            return

        if plt.isinteractive():
            plt.ioff()

        fig1, (ax5, ax6) = plt.subplots(1, 2, figsize=(10, 4))
        fig2, axes = plt.subplots(2, 2, figsize=(10, 10))
        ax7, ax8, ax9, ax10 = axes.ravel()

        C = pyebsd.list_cubic_symmetry_operators()

        # Calculate experimental (accurate?) OR
        filt = ipf.sel & (scan.CI > .2)
        V, V0, M_fcc, jvariants = pyebsd.OR_exp(M=scan.M, ph=scan.ph, sel=filt, C=C)

        Tracker.V = V
        Tracker.V0 = V0
        Tracker.M_fcc = M_fcc
        Tracker.jvariants = jvariants

        # C = list_cubic_symmetry_operators_KS()
        # KS = pyebsd.OR(C=C)
        Vr = pyebsd.average_orientation(V)

        # Calculate and plot map of the deviation from KS OR
        D_KS = np.dot(V, Vr.T)  # deviation from KS
        tr_KS = np.trace(D_KS, axis1=1, axis2=2)
        mis_KS = np.zeros(scan.N)
        mis_KS[filt & (scan.ph == 1)] = pyebsd.trace_to_angle(tr_KS)
        scan.plot_property(mis_KS, sel=filt & (scan.ph == 1),
                           vmin=0, vmax=5., gray=scan.IQ, tiling='hex', ax=ax5)

        Tracker.filt = filt
        Tracker.mis_KS = mis_KS

        # Plot crystallographic variants
        variants = np.zeros(scan.N)
        variants[filt & (scan.ph == 1)] = jvariants
        scan.plot_property(variants, sel=filt & (scan.ph == 1),
                           vmin=0, vmax=23, gray=scan.IQ, tiling='hex',
                           cmap=plt.get_cmap('jet'), ax=ax6)

        # Plot histogram of deviation from KS
        ax7.hist(mis_KS[filt & (scan.ph == 1)], bins=150)

        ax7.set_xlim(0, 10)
        ylim = ax7.get_ylim()

        mis_avg = np.mean(mis_KS[filt & (scan.ph == 1)])
        ax7.axvline(mis_avg, ls='--', color='k')
        ax7.annotate(u'Mean: {:.2g}°'.format(mis_avg),
                     xy=(mis_avg + .5, ylim[1]*.9),
                     ha='left')

        ax7.set_xlabel(u'Deviation from KS OR (°)')
        ax7.set_ylabel('Counts')

        print('Experimental')
        print('cpp: {:.4f} deg'.format(angle_2ds(V0, 'cpp')))
        print('cpd: {:.4f} deg'.format(angle_2ds(V0, 'cpd')))

        T = pyebsd.OR()[0]
        print('KS')
        print('cpp: {:.4f} deg'.format(angle_2ds(T, 'cpp')))
        print('cpd: {:.4f} deg'.format(angle_2ds(T, 'cpd')))

        T = pyebsd.OR(ds=([1, 1, 0], [1, 0, 0]))[0]
        print('NW')
        print('cpp: {:.4f} deg'.format(angle_2ds(T, 'cpp')))
        print('cpd: {:.4f} deg'.format(angle_2ds(T, 'cpd')))

        T = pyebsd.OR(ds=([5, 12, 17], [7, 17, 17]))[0]
        print('GT')
        print('cpp: {:.4f} deg'.format(angle_2ds(T, 'cpp')))
        print('cpd: {:.4f} deg'.format(angle_2ds(T, 'cpd')))

        ax8.hist(angle_2ds(V, 'cpp'), bins=150)
        ax8.hist(-angle_2ds(V, 'cpd'), bins=150)
        ax8.set_xlim(-8, 8)
        ax8.set_xlabel(('Deviation from closed packed direction (-) '
                        'or closed packed plane (+) (deg)'))
        ax8.set_ylabel('Counts')

        proj0, proj = [1, 1, 0], [1, 1, 1]

        scan.plot_PF(proj=proj0, sel=filt & (scan.ph == 2), rotation=M_fcc.T,
                     contour=True, cmap='Blues_r', bins=512, nlevels=20, ax=ax9)
        pyebsd.plot_PF(M=V, contour=True, cmap='Reds_r',
                       bins=512, nlevels=20, fn=None, proj=proj, ax=ax9)
        pyebsd.plot_PF(M=np.dot(V0, C).transpose([1, 0, 2]),
                       mfc=[1, 1, 1, 0], mec='k', mew=1, marker='o',
                       ms=3, proj=proj, verbose=False, ax=ax9)
        ax9.axis('off')
        ax9.set_aspect('equal')
        pyebsd.draw_circle_frame(ax9, lw=.5)
        pyebsd.draw_std_traces(ax9, lw=.2)
        pyebsd.draw_wulff_net(ax9, lw=.2)

        #####################

        parent_or = np.diag([1, 1, 1])
        # axis0, axis1 = np.asarray(parent_or)[:, 0], np.asarray(parent_or)[:, 1]

        T_KS = pyebsd.OR()  # K-S
        T_NW = pyebsd.OR(ds=[[1, 1, 0], [1, 0, 0]])  # N-W
        T_GT = pyebsd.OR(ds=[[5, 12, 17], [7, 17, 17]])  # G-T

        proj = [1, 0, 0]

        pyebsd.plot_PF(M=np.dot(V0, C).transpose([1, 0, 2]),
                       mfc=[1, 1, 1, 0], mec='k', mew=1, marker='o',
                       ms=5, proj=proj, ax=ax10, label='exp.')
        pyebsd.plot_PF(M=T_KS, mfc=[1, 1, 1, 0], mec='r', mew=1,
                       marker='s', ms=5, proj=proj, rotation=parent_or,
                       verbose=False, ax=ax10, label='KS')
        pyebsd.plot_PF(M=T_NW, mfc=[1, 1, 1, 0], mec='g', mew=1,
                       marker='D', ms=5, proj=proj, rotation=parent_or,
                       verbose=False, ax=ax10, label='NW')
        pyebsd.plot_PF(M=T_GT, mfc=[1, 1, 1, 0], mec='b', mew=1,
                       marker='v', ms=5, proj=proj, rotation=parent_or,
                       verbose=False, ax=ax10, label='GT')
        ax10.axis('off')
        ax10.set_aspect('equal')
        pyebsd.draw_circle_frame(ax10, lw=.5)
        pyebsd.draw_std_traces(ax10, lw=.2)
        pyebsd.draw_wulff_net(ax10, lw=.2)
        ax10.legend()

        plt.ion()
        plt.show()

    def on_key_press(event):
        if event.key == 'shift':
            if np.count_nonzero(ipf.sel) == 0:
                return

            M_fcc = pyebsd.average_orientation(scan.M,
                                               sel=ipf.sel & (scan.ph == 2) & (scan.CI > .2),
                                               vectorized=False, verbose=True)

            # plot IPF of the selected data
            ax2.cla()  # clear axis
            scan.plot_IPF(sel=ipf.sel, gray=scan.IQ, tiling='hex', ax=ax2)

            # plot PF of the selected data
            ax4.cla()  # clear axis
            ax4.set_aspect('equal')
            ax4.axis('off')
            pyebsd.draw_circle_frame(ax4, lw=.5)
            pyebsd.draw_std_traces(ax4, lw=.5)
            scan.plot_PF(sel=ipf.sel & (scan.ph == 1) & (scan.CI > .2), proj=proj,
                         contour=True, cmap=plt.get_cmap('Reds_r'), ax=ax4, rotation=M_fcc.T)
            scan.plot_PF(sel=ipf.sel & (scan.ph == 2) & (scan.CI > .2), proj=proj,
                         contour=True, cmap=plt.get_cmap('Blues_r'), ax=ax4, rotation=M_fcc.T)

            plt.draw_all()

        elif event.key == 'enter':
            calculate_accurate_OR()

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    # fig.canvas.mpl_connect('close_event', calculate_accurate_OR)

    # plt.close('all')

    # if np.count_nonzero(ipf.sel) > 0:
    #     sel = ipf.sel
    #     exec(open('accurate_OR.py').read())
