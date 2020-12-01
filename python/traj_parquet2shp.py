import time

import cudf
import cuspatial
import numpy as np

from osgeo import ogr
from shapely.geometry import LineString

start = time.time()
df = cudf.read_parquet("/home/jianting/trajectorysample/pplTraj_2020-08-01.parquet")
end = time.time()
print('parquet file reading time :', (end-start)*1000)
print('number of rows:',len(df))    

#df.query('length==1')
df=df.query('length>1')

#df.head().to_pandas().columns
locs= df['locations'].to_array()
lengths=df['length'].to_array()

start = time.time()
driver = ogr.GetDriverByName('Esri Shapefile')
ds = driver.CreateDataSource('traj_test.shp')
layer = ds.CreateLayer('', None, ogr.wkbLineString)
#layer.CreateField(ogr.FieldDefn('numVer', ogr.OFTInteger))
defn = layer.GetLayerDefn()

for i in range(len(locs)): 
 coor=eval(locs[i])
 
 #not needed after cudf query on GPUs
 '''
 if(len(coor)<2):
  print(i," ",coor," ",locs[i])
  continue
 '''
 
 lr=LineString(coor)
 assert(len(lr.coords)==lengths[i])
 feat = ogr.Feature(defn)
 geom = ogr.CreateGeometryFromWkb(lr.wkb)
 feat.SetGeometry(geom)
 #feat.SetField('numVer', lengths[i])
 layer.CreateFeature(feat)
 fea=geom=None
layer=ds=None
end = time.time()

print('number of trajectories:',len(locs))    
print('trajectory parsing time :', (end-start)*1000)


