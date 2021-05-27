from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyebsd

if __name__ == '__main__':
    nrows = 10
    ncols = 10

    width = ncols // 5
    height = nrows // 5

    dx = .1

    y, x = np.mgrid[:nrows, :ncols]

    rawdata = pd.DataFrame(index=range(nrows * ncols))
    rawdata['phi1'] = 2 * np.pi
    rawdata['Phi'] = 2 * np.pi
    rawdata['phi2'] = 2 * np.pi
    rawdata['x'] = x.ravel()
    rawdata['y'] = y.ravel()
    rawdata['ph'] = -1
    rawdata['IQ'] = 1
    rawdata['CI'] = 1
    rawdata['intensity'] = 1
    rawdata['fit'] = 1
    rawdata['variant'] = -1

    euler_angles_fcc = [0.1, 1.2, 1.3]
    M_fcc = pyebsd.euler_angles_to_rotation_matrix(*euler_angles_fcc).T
    T_KS = pyebsd.OR(C=pyebsd.list_cubic_symmetry_operators_KS())
    M_bcc = np.matmul(T_KS, M_fcc)

    euler_angles_bcc = pyebsd.rotation_matrix_to_euler_angles(M_bcc.transpose(0, 2, 1))
    cy_euler_angles_bcc = cycle(zip(*euler_angles_bcc))

    for i in range(5):
        for j in range(5):
            window = (rawdata.x >= j*width) & (rawdata.x < (j+1)*width)
            window &= (rawdata.y >= i*height) & (rawdata.y < (i+1)*height)

            if (i == 2) and (j == 2):
                phi1, Phi, phi2 = euler_angles_fcc
                ph = 2
            else:
                phi1, Phi, phi2 = next(cy_euler_angles_bcc)
                ph = 1
                rawdata.loc[window, 'variant'] = 5*i + j

            rawdata.loc[window, 'phi1'] = phi1
            rawdata.loc[window, 'Phi'] = Phi
            rawdata.loc[window, 'phi2'] = phi2
            rawdata.loc[window, 'ph'] = ph

    # rawdata['phi1'] = phi1
    # rawdata['Phi'] = Phi
    # rawdata['phi2'] = phi2

    rawdata['x'] *= dx
    rawdata['y'] *= dx

    scan = pyebsd.ScanData(rawdata, 'SqrGrid', dx, dx, ncols, ncols, nrows)

    V, Vavg, Mprt, variants = pyebsd.OR_exp(
        scan.M, scan.ph, C=pyebsd.list_cubic_symmetry_operators_KS())

    variants2 = np.full(scan.N, -1)
    variants2[scan.ph == 1] = variants

    scan.plot_property(variants2)

    plt.ion()
    plt.show()
