import os
import cupy
import cudf
import cuspatial
import numpy as np
 
cuspatial_data_path='/home/microway/cuspatial_data' 
polygon_shapefile_name='taxi_zones.shp'
#polygon_shapefile_name='nycd_11a_av/nycd.shp'
#polygon_shapefile_name='nyct2000_11a_av/nyct2000.shp'

def read_points(path):
    print('reading points file:', path)
    points = np.fromfile(path, dtype=np.int32)
    points = cupy.asarray(points)
    points = points.reshape((len(points)// 4, 4))
    points = cudf.DataFrame.from_gpu_matrix(points)
    return points

points_df = cudf.concat(
    [read_points(os.path.join(cuspatial_data_path,
        '2009{}.cny'.format('0{}'.format(i) if i < 10 else i)))
    for i in range(1, 13)]).reset_index(drop=True)
points_df['x'] = points_df[0].astype(np.float32)
points_df['y'] = points_df[1].astype(np.float32)
print(len(points_df))

ply_fpos, ply_rpos, ply_vertices = cuspatial.read_polygon_shapefile(
    os.path.join(cuspatial_data_path , polygon_shapefile_name)) 
x1,x2,y1,y2 =(ply_vertices['x'].min(), ply_vertices['x'].max(), 
    ply_vertices['y'].min(),  ply_vertices['y'].max())
                                 
num_levels = 15
min_size = 512
scale = max(abs(x2 - x1), abs(y2 - y1)) / ((1 << num_levels) - 2);

key_to_point,quadtree = cuspatial.quadtree_on_points(points_df['x'],points_df['y'],
    x1,x2,y1,y2, scale,num_levels, min_size)
print(len(quadtree))

print('node count:', (quadtree['is_quad'] == True).sum())
print('leaf count:', (quadtree['is_quad'] == False).sum())

poly_bboxes = cuspatial.polygon_bounding_boxes(
    ply_fpos, ply_rpos, ply_vertices['x'], ply_vertices['y'])
print(len(poly_bboxes))    

intersections = cuspatial.join_quadtree_and_bounding_boxes(
    quadtree, poly_bboxes,x1,x2,y1,y2, scale, num_levels,
)
print(len(intersections))    

polygons_and_points = cuspatial.quadtree_point_in_polygon(
    intersections, quadtree,key_to_point,
    points_df['x'],points_df['y'],
    ply_fpos,ply_rpos,ply_vertices['x'],ply_vertices['y'])
print(len(polygons_and_points)) 

