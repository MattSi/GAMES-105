import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA


p2 = np.array([0.173241,  -0.173241,   0.000000])
#p2 = np.array([0.245000, 0,0])
p1 = np.array([0.245000,  0.0,   0.000000])

r = R.from_euler('XYZ', [0,0,-45], degrees=True)
print(r.as_matrix().dot(p1))

r_axis = np.cross(p2, p1)
n_r_axis = np.linalg.norm(r_axis)
u = r_axis / n_r_axis
theta = np.arccos(np.dot(p1,p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))
eula = R.from_rotvec(u*theta).as_euler('XYZ', degrees=True)

print(eula)

