import numpy as np
import cudf
import cuspatial

np.random.seed(0)
points = cudf.DataFrame({
        "x": cudf.Series(np.random.normal(size=120)) * 500,
        "y": cudf.Series(np.random.normal(size=120)) * 500,
    })

max_depth = 3
min_size = 50
min_x, min_y, max_x, max_y = (points["x"].min(),
                                  points["y"].min(),
                                  points["x"].max(),
                                  points["y"].max())
scale = max(max_x - min_x, max_y - min_y)/ ((1 << max_depth)-2)
print(
        "min_size:   " + str(min_size) + "\n"
        "num_points: " + str(len(points)) + "\n"
        "min_x:      " + str(min_x) + "\n"
        "max_x:      " + str(max_x) + "\n"
        "min_y:      " + str(min_y) + "\n"
        "max_y:      " + str(max_y) + "\n"
        "scale:      " + str(scale) + "\n"
    )


key_to_point, quadtree = cuspatial.quadtree_on_points(
        points["x"],
        points["y"],
        min_x,
        max_x,
        min_y,
        max_y,
        scale, max_depth, min_size
    )
print(quadtree)
print(key_to_point)