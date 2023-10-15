import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA


p1 = np.array([0.173241,-0.173241,0])
p2 = np.array([0.245000 ,0,0])

r = R.from_euler('XYZ', [0,0,-45], degrees=True)
print(r.as_matrix().dot(p2))

print(r.as_matrix())
print(r.as_quat())
print(r.as_rotvec())


