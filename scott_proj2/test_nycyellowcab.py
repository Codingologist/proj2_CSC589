import sys
import numpy as np
sys.path.append('./build')

import pandas as pd
import timeit
#import cudf
import haversine_library

#code from: https://github.com/rapidsai/cuspatial/blob/724d170a2105441a3533b5eaf9ee82ddcfc49be0/notebooks/nyc_taxi_years_correlation.ipynb
#data from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

taxis = []
#taxi = cudf.read_parquet("yellow_tripdata_2009-01.parquet")

for i in range(1,13):
    taxi = pd.read_parquet(f"yellow_tripdata_2009/yellow_tripdata_2009-{i}.parquet")
    taxi = taxi[(taxi['Start_Lon'] >= -74.15) & (taxi['Start_Lat'] >= 40.5774) & (taxi['End_Lon'] <= -73.7004) & (taxi['End_Lat'] <= 40.9176)]
    taxis.append(taxi)

final_df = pd.concat(taxis, ignore_index=True)

def haversine_distance_py(size, x1, y1, x2, y2, dist):
        R = 6371
        for i in range(size):
                dLat = (y2[i] - y1[i]) * (np.pi / 180)
                dLon = (x2[i] - x1[i]) * (np.pi / 180)
                y1Rad = y1[i] * (np.pi / 180)
                y2Rad = y2[i] * (np.pi / 180)
                a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(y1Rad) * np.cos(y2Rad) * np.sin(dLon / 2) * np.sin(dLon/2)
                c = 2 * np.atan2(np.sqrt(a), np.sqrt(1-a))
                dist[i] = R * c

x1=taxi['Start_Lon'].to_numpy()
y1=taxi['Start_Lat'].to_numpy()
x2=taxi['End_Lon'].to_numpy()
y2=taxi['End_Lat'].to_numpy()
size=len(x1)
dist=np.zeros(size)
start_time = timeit.default_timer()
haversine_library.haversine_distance(size,x1,y1,x2,y2,dist)
end_time = timeit.default_timer()
diff = end_time - start_time
print("GPU accelerated time (timeit): ", diff, " secs")

start_time_python = timeit.default_timer()
haversine_distance_py(size, x1, y1, x2, y2, dist)
end_time_python = timeit.default_timer()
diff_py = end_time_python - start_time_python
print("Python Implementation Time (timeit): ", diff_py, " secs")

print("GPU accelerated time has CPU beat by: ", diff_py - diff, " secs")