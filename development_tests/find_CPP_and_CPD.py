import numpy as np
import pyebsd

if __name__ == '__main__':
    C = pyebsd.list_cubic_symmetry_operators_KS()
    T_KS = pyebsd.OR(C=C)

    FCC_CPP = (np.array([1, 1, 1])/3.**.5)
    FCC_CPD = (np.array([1, 1, 0])/2.**.5)

    for var, T in enumerate(T_KS):
        if var % 2 == 1:
            T = C[15].dot(T)
        TC = np.matmul(T, C)

        bcc_cpp_C = np.abs(TC.dot(FCC_CPP))
        bcc_cpd_C = np.abs(TC.dot(FCC_CPD))

        bcc_cpp_C /= bcc_cpp_C.max(axis=1).reshape(-1, 1)
        bcc_cpd_C /= bcc_cpd_C.max(axis=1).reshape(-1, 1)

        bcc_cpp_C = bcc_cpp_C.round(6)
        bcc_cpd_C = bcc_cpd_C.round(6)

        idx_cpp, = np.where(np.count_nonzero(bcc_cpp_C == 1, axis=1) == 2)
        idx_cpd, = np.where(np.count_nonzero(bcc_cpd_C == 1, axis=1) == 3)

        fcc_cpp = C[idx_cpp[0]].dot(FCC_CPP)
        fcc_cpd = C[idx_cpd[0]].dot(FCC_CPD)

        bcc_cpp = T.dot(fcc_cpp)
        bcc_cpd = T.dot(fcc_cpd)

        fcc_cpp = (fcc_cpp/np.abs(fcc_cpp).max()).round(6).astype(int)
        fcc_cpd = (fcc_cpd/np.abs(fcc_cpd).max()).round(6).astype(int)

        bcc_cpp = (bcc_cpp/np.abs(bcc_cpp).max()).round(6).astype(int)
        bcc_cpd = (bcc_cpd/np.abs(bcc_cpd).max()).round(6).astype(int)

        string = '{:2d}'.format(var + 1)
        string += '  |  [{:2d} {:2d} {:2d}] // [{:2d} {:2d} {:2d}]'.format(*fcc_cpp, *bcc_cpp)
        string += '  |  [{:2d} {:2d} {:2d}] // [{:2d} {:2d} {:2d}]'.format(*fcc_cpd, *bcc_cpd)
        print(string)

        if (var+1) % 6 == 0 and var + 1 != len(T_KS):
            print('----+----------------------------+--------------------------')
