import numpy as np

file = np.load('./sim_time_2.npz')
# Time = file["first_route_conn_time"]
Time = file["second_route_conn_time"]
# Time = file["full_conn_time"]
Time = Time.transpose()
n = len(Time)

min_T = 0
for iter_idx in range(1, n):
    temp = []
    temp.extend(np.diagonal(Time, offset=iter_idx))
    temp.extend(np.diagonal(Time, offset=iter_idx-n))
    min_T = min_T + max(temp)
print(min_T)