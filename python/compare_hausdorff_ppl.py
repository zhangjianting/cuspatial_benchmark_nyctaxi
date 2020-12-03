import time

import cudf
import cuspatial

start = time.time()
df = cudf.read_parquet("/home/jianting/trajectorysample/pplTraj_2020-08-01.parquet")
end = time.time()
print('parquet file reading time :', (end-start)*1000)
print('number of rows:',len(df))    

df=df.query('length>1')

#df.head().to_pandas().columns
locs= df['locations']
lengths=df['length']

start = time.time()
s1 = locs.str.replace(['[','(',')',']'], [''], regex=False)
t1 = s1.str.tokenize(',').str.strip()
d1 = t1.astype('double')
xs= d1[0:len(d1):2]
ys = d1[1:len(d1):2]
end = time.time()
num_traj=len(locs)
print('number of trajectories:',num_traj)    
print('trajectory parsing time :', (end-start)*1000)

#OOM error on Titan V 12 GB beyond this limit (14000)
ul=1000
start = time.time()
cuspatial_dist=cuspatial.directed_hausdorff_distance(xs[0:ul],ys[0:ul],lengths[0:ul])
cuspatial_dist=cuspatial_dist.as_matrix()
end = time.time()
# ~800 ms on Titan V 12GB
print('directed_hausdorff_distance time :', (end-start)*1000)

pnt_x=xs.to_array()
pnt_y=ys.to_array()

start = time.time()
trajs = []
c = 0
for i in range(num_traj):
    traj = np.zeros((n[i], 2), dtype=np.float64)
    for j in range(n[i]):
        traj[j][0] = pnt_x[c + j]
        traj[j][1] = pnt_y[c + j]
    trajs.append(traj.reshape(-1, 2))
    c = c + n[i]
# print('c={}'.format(c))
end = time.time()
print("CPU traj prep time={}".format((end - start) * 1000))
# print(trajs[0]=",trajs[0])

mis_match = 0
d = np.zeros((num_traj, num_traj), dtype=np.float64)
for i in range(num_traj):
    if i % 100 == 99:
        print("i={}".format(i))
    for j in range(num_traj):
        dij = directed_hausdorff(trajs[i], trajs[j])
        d[i][j] = dij[0]
        if abs(d[i][j] - cuspatial_dist[i][j]) > 0.00001:
            print("{} {} {} {}".format(i, j, d[i][j], cuspatial_dist[i][j]))
            mis_match = mis_match + 1
print("mis_match={}".format(mis_match))
end = time.time()

print(
    "python Directed Hausdorff distance cpu end-to-end time in ms "
    "(end-to-end)={}".format((end - start) * 1000)
)
