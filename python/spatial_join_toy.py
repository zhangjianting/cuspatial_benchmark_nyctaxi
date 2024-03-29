import numpy as np
import cudf
import cuspatial

small_poly_offsets = cudf.Series([0, 1, 2, 3], dtype=np.uint32)

small_ring_offsets = cudf.Series([0, 3, 8, 12], dtype=np.uint32)

small_poly_xs = cudf.Series([
2.488450,1.333584,3.460720,5.039823,5.561707,7.103516,7.190674,5.998939,5.998939,5.573720,6.703534,5.998939,2.088115,1.034892,2.415080,3.208660,2.088115])  

small_poly_ys = cudf.Series([
5.856625,5.008840,4.586599,4.229242,1.825073,1.503906,4.025879,5.653384,1.235638,0.197808,0.086693,1.235638,4.541529,3.530299,2.896937,3.745936,4.541529]) 

small_points_x = cudf.Series([
1.9804558865545805,0.1895259128530169,1.2591725716781235,0.8178039499335275,0.48171647380517046,1.3890664414691907,0.2536015260915061,3.1907684812039956,3.028362149164369,3.918090468102582,3.710910700915217,3.0706987088385853,3.572744183805594,3.7080407833612004,3.70669993057843,3.3588457228653024,2.0697434332621234,2.5322042870739683,2.175448214220591,2.113652420701984,2.520755151373394,2.9909779614491687,2.4613232527836137,4.975578758530645,4.07037627210835,4.300706849071861,4.5584381091040616,4.822583857757069,4.849847745942472,4.75489831780737,4.529792124514895,4.732546857961497,3.7622247877537456,3.2648444465931474,3.01954722322135,3.7164018490892348,3.7002781846945347,2.493975723955388,2.1807636574967466,2.566986568683904,2.2006520196663066,2.5104987015171574,2.8222482218882474,2.241538022180476,2.3007438625108882,6.0821276168848994,6.291790729917634,6.109985464455084,6.101327777646798,6.325158445513714,6.6793884701899,6.4274219368674315,6.444584786789386,7.897735998643542,7.079453687660189,7.430677191305505,7.5085184104988,7.886010001346151,7.250745898479374,7.769497359206111,1.8703303641352362,1.7015273093278767,2.7456295127617385,2.2065031771469,3.86008672302403,1.9143371250907073,3.7176098065039747,0.059011873032214,3.1162712022943757,2.4264509160270813,3.154282922203257])

small_points_y = cudf.Series([
1.3472225743317712,0.5431061133894604,0.1448705855995005,0.8138440641113271,1.9022922214961997,1.5177694304735412,1.8762161698642947,0.2621847215928189,0.027638405909631958,0.3338651960183463,0.9937713340192049,0.9376313558467103,0.33184908855075124,0.09804238103130436,0.7485845679979923,0.2346381514128677,1.1809465376402173,1.419555755682142,1.2372448404986038,1.2774712415624014,1.902015274420646,1.2420487904041893,1.0484414482621331,0.9606291981013242,1.9486902798139454,0.021365525588281198,1.8996548860019926,0.3234041700489503,1.9531893897409585,0.7800065259479418,1.942673409259531,0.5659923375279095,2.8709552313924487,2.693039435509084,2.57810040095543,2.4612194182614333,2.3345952955903906,3.3999020934055837,3.2296461832828114,3.6607732238530897,3.7672478678985257,3.0668114607133137,3.8159308233351266,3.8812819070357545,3.6045900851589048,2.5470532680258002,2.983311357415729,2.2235950639628523,2.5239201807166616,2.8765450351723674,2.5605928243991434,2.9754616970668213,2.174562817047202,3.380784914178574,3.063690547962938,3.380489849365283,3.623862886287816,3.538128217886674,3.4154469467473447,3.253257011908445,4.209727933188015,7.478882372510933,7.474216636277054,6.896038613284851,7.513564222799629,6.885401350515916,6.194330707468438,5.823535317960799,6.789029097334483,5.188939408363776,5.788316610960881]) 

dtype=np.float32
x_min = 0
x_max = 8
y_min = 0
y_max = 8
scale = 1
max_depth = 3
min_size = 12
points_x = small_points_x.astype(dtype)
points_y = small_points_y.astype(dtype)
poly_points_x = small_poly_xs.astype(dtype)
poly_points_y = small_poly_ys.astype(dtype)

point_indices, quadtree = cuspatial.quadtree_on_points(
    points_x,points_y,x_min,x_max,y_min,y_max,scale,max_depth,min_size)

poly_bboxes = cuspatial.polygon_bounding_boxes(
    small_poly_offsets, small_ring_offsets, poly_points_x, poly_points_y)

intersections = cuspatial.join_quadtree_and_bounding_boxes(
    quadtree, poly_bboxes, x_min, x_max, y_min, y_max, scale, max_depth)

polygons_and_points = cuspatial.quadtree_point_in_polygon(
    intersections,quadtree,point_indices,points_x,points_y,small_poly_offsets,small_ring_offsets,poly_points_x,poly_points_y)

#ply_idx are squentially numbered
#pnt_idx are offsets into point_indices
ply_idx= polygons_and_points['polygon_index']
pnt_idx= polygons_and_points['point_index']
 
#polygon_index [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]
#point_index [62,60,45,46,47,48,49,50,51,52,54,28,29,30,31,32,33,34,35]

'''
total_points=len(points_x)
df=cudf.DataFrame({'a':point_indices})
idx=cudf.core.column.as_column(np.arange(total_points), dtype="uint32")
val=df.take(pnt_idx)
diff=point_indices._column-val['a']._column;
'''

import shapefile
from shapely.geometry import Point, Polygon

plyreader = shapefile.Reader('/home/geoteci/cuspatial_data/quad_test_ply.shp')
polygon = plyreader.shapes()

plys = []
for shape in polygon:
    plys.append(Polygon(shape.points))

np_pnt_x=points_x.to_numpy()
np_pnt_y=points_y.to_numpy()

#verify for each points in the input point arrays
total_points=len(points_x)
for i in range(total_points):
    k=point_indices[i]
    pt = Point(np_pnt_x[k], np_pnt_y[k])
    for j in range(len(plys)):
        pip = plys[j].contains(pt)
        if(pip):
           print(i,'...',k,'-->',j)

#verify for each matched point/polygon pair
num_points=len(pnt_idx)
for i in range(num_points):
    #pnt_idx has offsets to point_indices; point_indices has offsets to the orginal input point array
    m=pnt_idx[i]
    n=point_indices[m]
    pt = Point(np_pnt_x[n], np_pnt_y[n])
    for j in range(len(plys)):
        pip = plys[j].contains(pt)
        if(pip):
            print(i,'...',m,'...',n,'-->',j,'|',ply_idx[i])

#verify that non-matched points are outside of any polygons
match_idx=[point_indices[pnt_idx[i]] for i in range(num_points)]
non_match_idx=np.setdiff1d(np.arange(total_points),match_idx)
num_error=0
for i in range(len(non_match_idx)):
    k=non_match_idx[i]
    pt = Point(np_pnt_x[k], np_pnt_y[k])
    for j in range(len(plys)):
       if(plys[j].contains(pt)):
           num_error=num_error+1
#num_error should be zero
print('num_error=',num_error)