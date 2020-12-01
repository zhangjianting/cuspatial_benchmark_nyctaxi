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
print('number of trajectories:',len(locs))    
print('trajectory parsing time :', (end-start)*1000)

#OOM error on Titan V 12 GB beyond this limit
ul=14000
start = time.time()
sim=cuspatial.directed_hausdorff_distance(xs[0:ul],ys[0:ul],lengths[0:ul])
end = time.time()
print('number of similarities:',len(sim))    
# ~800 ms on Titan V 12GB
print('directed_hausdorff_distance time :', (end-start)*1000)


