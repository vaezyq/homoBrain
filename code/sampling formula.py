"""
流量估计采样计算公式
"""

neuron_number = 8.64e10


conn_number_esti[dst, src] = neuron_number * size[dst] * degree[dst] * conn[dst, src]

conn_number[dsts, src] = np.unique(np.choice(neuron_number * size[src],
                                   sum([conn_number_esti[d, src] for d in dsts]),
                                   replace=True)).shape[0]

conn_number[dsts, srcs] = sum([conn_number[dsts, s] for s in srcs])
