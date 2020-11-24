import time

from cudf import Series, read_csv

import cuspatial

start = time.time()
# data dowloaded from
# https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2009-01.csv

df = read_csv("/home/jianting/cuspatial_data/yellow_tripdata_2009-01.csv")
end = time.time()
print("data ingesting time (from SSD) in ms={}".format((end - start) * 1000))

print(df.head().to_pandas().columns)

#save a suset of four columns for tests using java and python on jetson
subset=df[['Start_Lon','Start_Lat','End_Lon','End_Lat']]
subset.to_parquet("/home/jianting/cuspatial_data/200901.parquet")

start = time.time()
x1 = Series(df["Start_Lon"])
y1 = Series(df["Start_Lat"])
x2 = Series(df["End_Lon"])
y2 = Series(df["End_Lat"])
end = time.time()
print(
    "data frame to column conversion time in ms={}".format(
        (end - start) * 1000
    )
)
start = time.time()
h_dist = cuspatial.haversine_distance(x1, y1, x2, y2)
end = time.time()
print("python computing distance time in ms={}".format((end - start) * 1000))
# h_dist.data.to_array()
