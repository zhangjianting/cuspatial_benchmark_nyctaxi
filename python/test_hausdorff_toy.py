import numpy as np
import cudf
import cuspatial
from scipy.spatial.distance import directed_hausdorff

in_trajs = []
in_trajs.append(np.array([[1, 0], [2, 1], [3, 2], [5, 3], [7, 1]]))
in_trajs.append(np.array([[0, 3], [2, 5], [3, 6], [6, 5]]))
in_trajs.append(np.array([[1, 4], [3, 7], [6, 4]]))
out_trajs = np.concatenate([np.asarray(traj) for traj in in_trajs], 0)

traj_x = np.array(out_trajs[:, 0],dtype=np.float64)
traj_y = np.array(out_trajs[:, 1],dtype=np.float64)
traj_offset = []
traj_cnt = []
traj_init=0
for traj in in_trajs:
    traj_cnt.append(len(traj))
    traj_offset.append(traj_init)
    traj_init=traj_init+len(traj)

pnt_x = cudf.Series(traj_x)
pnt_y = cudf.Series(traj_y)

#for release >=0.15 
offset = cudf.Series(traj_offset)
cuspatial_dist = cuspatial.directed_hausdorff_distance(pnt_x, pnt_y, offset)

#for release <=0.14 
#cnt= cudf.Series(traj_cnt)
#cuspatial_dist = cuspatial.directed_hausdorff_distance(pnt_x, pnt_y, cnt)

cuspatial_dist=cuspatial_dist.as_matrix()
print(cuspatial_dist)

num_traj = len(traj_offset)
mis_match = 0
d = np.zeros((num_traj, num_traj), dtype=np.float64)
for i in range(num_traj):
    for j in range(num_traj):
        dij = directed_hausdorff(in_trajs[i], in_trajs[j])
        d[i][j] = dij[0]
        if abs(d[i][j] - cuspatial_dist[i][j]) > 0.00001:
            print("{} {} {} {}".format(i, j, d[i][j], cuspatial_dist[i][j]))
            mis_match = mis_match + 1

print("mis_match={}".format(mis_match))

