import numpy as np
from scipy.spatial.transform import Rotation
import pyebsd

# Random orientations
ea1 = [4.13273, 0.37537, 1.84485]
M1 = pyebsd.euler_angles_to_rotation_matrix(*ea1, verbose=False)
R1 = Rotation.from_euler('zxz', ea1)

ea2 = [1.27017, 1.21478, 6.20018]
M2 = pyebsd.euler_angles_to_rotation_matrix(*ea2, verbose=False)
R2 = Rotation.from_euler('zxz', ea2)

# Symmetry operators
C = pyebsd.list_cubic_symmetry_operators()

# Empty array where the misorientation values will be stored
mis = np.zeros((len(C), len(C)))
# Empty array where the misorientation values calculated with
# the quaternions will be stored
mis_q = np.zeros((len(C), len(C)))

# Loop over first set of symmetry operators (Ci)
for i in range(len(C)):
    # Loop over second set of symmetry operators (Cj)
    for j in range(len(C)):
        A = C[i].dot(M1)
        B = C[j].dot(M2)

        # Rotation matrix describing rotation from A to B
        Mis = A.dot(B.T)
        # Misorientation angle calculated from the trace of Mis
        mis[i, j] = np.arccos((Mis.trace() - 1.)/2.).round(8)

        # qA = (Rotation.from_dcm(C[i])*R1).as_quat()
        # qB = (Rotation.from_dcm(C[j])*R2).as_quat()
        # mis_q[i, j] = 2.*np.arccos((np.dot(qA, qB) - 1.)/2.).round(8)

# Use set to remove repeated values and sort them
mis0 = set(mis[:, 0])
for j in range(len(C)):
    # Check if multiplying M2 with Cj is reduntant or not
    # by checking if set(mis[:, j]) is equal to set(mis[:, 0])
    print(j, mis0 == set(mis[:, j]))

# with open('misorientation_calc_test.txt', 'w') as file:
#     print(mis, file=file)
