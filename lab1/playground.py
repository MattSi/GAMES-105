import numpy as np
from scipy.spatial.transform import Rotation as R


p1 = np.array([0,1,0])
p2 = np.array([0,1,0])

u = p1/np.linalg.norm(p1)
v = p2/np.linalg.norm(p2)

axis = np.cross(u,v)
angle = np.arccos(np.dot(u,v))

q = R.from_rotvec(angle * axis)

print(q.as_quat())