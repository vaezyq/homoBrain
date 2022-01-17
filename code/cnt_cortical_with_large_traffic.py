import numpy as np
import pickle
import time

path = '../tables/traffic_table/'
file_name = "traffic_base_cortical_map_10000_v1_cortical_v2.pkl"

map_path = "../tables/map_table/map_10000_v1_cortical_v2_without_invalid_idx.pkl"

N = 10000
n = 171508

print(file_name)

time1 = time.time()
traffic_base_cortical_path = path + file_name

with open(traffic_base_cortical_path, 'rb') as f:
    traffic_base_cortical = pickle.load(f)
print(traffic_base_cortical_path + ' loaded.')
time2 = time.time()
print("%.2f seconds consumed." % (time2 - time1))

with open(map_path, 'rb') as f:
    map_table = pickle.load(f)

print("map table loaded.")

time1 = time.time()
traffic_per_cortical = np.zeros(n)

for gpu_idx in range(N):
    for dst_idx in range(N):
        if gpu_idx != dst_idx:
            for i in range(len(map_table[str(gpu_idx)])):
                cortical_idx = map_table[str(gpu_idx)][i]
                traffic_per_cortical[cortical_idx] += traffic_base_cortical[dst_idx][gpu_idx][i]
    if gpu_idx % 100 == 0:
        print(gpu_idx)
time2 = time.time()
print('%.2f' % (time2 - time1))

max_times = 10

lst = np.zeros(max_times + 1, dtype=int)
average = np.sum(traffic_per_cortical) / 10000
for i in range(n):
    for j in range(1, max_times + 1, 1):
        if traffic_per_cortical[i] >= j * average:
            lst[j] += 1

for i in range(1, max_times + 1, 1):
    print("%d: %d" % (i, lst[i]))
