import numpy as np
import pandas as pd
import pyebsd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Symmetry operators sorted according to the KS variants convention
    C = pyebsd.list_cubic_symmetry_operators_KS()
    KS = pyebsd.OR(ps=[[1, 1, 1], [0, 1, 1]], ds=[[-1, 0, 1], [-1, -1, 1]],
                   single=True)  # only the first matrix

    # The orientations in M_KS are in the order of the variants convetion
    # published in papers such as:
    # T. Furuhara, T. Chiba, T. Kaneshita, H. Wu, G. Miyamoto, Metall.
    # Mater. Trans. A 48 (2017) 2739–2752.
    M_KS = np.matmul(KS, C.transpose(0, 2, 1))  # KS * C^T

    v = 0  # variant
    variants_found = set()
    variants = pd.DataFrame(index=range(len(M_KS)))
    variants['Variant number'] = variants.index + 1
    variants['CP group'] = np.repeat(range(4), 6)
    variants['Bain group'] = -1
    for group in range(3):
        misang = [pyebsd.misorientation(M_KS[j], M_KS[v]) for j in range(len(M_KS))]

        if v == 0:
            variants['Rotation from variant 1 (deg)'] = misang

        mis = pd.DataFrame(dict(vno=range(24), mis=misang))
        mis.sort_values(by='mis', inplace=True)
        mis.reset_index(inplace=True, drop=True)

        # Once sorted, the 8 first variants belong to the same Bain group
        variants_grouped = mis['vno'][:8].values

        variants.loc[variants_grouped, 'Bain group'] = group
        variants_found.update(variants_grouped)

        v = 8
        while v in variants_found:
            v += 1

    variants['CP group'] += 1
    variants['Bain group'] += 1
    print(variants)
