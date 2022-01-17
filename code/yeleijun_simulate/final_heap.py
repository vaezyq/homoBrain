import heapq
import numpy as np
from collections import Counter
import time
import random
from multiprocessing import Pool, Lock


def main(Time, epoch, rand_flag=False, wait_flag=True, all2all_flag=False):
    class linked_node():
        def __init__(self, node):
            self.linked_pair = []
            heapq.heappush(self.linked_pair, node)

        def add_link(self, node):
            heapq.heappush(self.linked_pair, node)

        def remove_link(self):
            return heapq.heappop(self.linked_pair)

        def print_link(self):
            print(self.linked_pair)

        def min_time_link(self):
            return self.linked_pair[0][0]

        def send_node(self):
            return [self.linked_pair[k][1] for k in range(len(self.linked_pair))]

        def recieve_node(self):
            return [self.linked_pair[k][2] for k in range(len(self.linked_pair))]

    print(epoch)
    Time = Time.transpose()
    n = len(Time)

    nonzero_idx = Time.nonzero()
    dict_count = dict(Counter(nonzero_idx[0]))
    cun_count = np.cumsum(list(dict_count.values()))
    stratage = [list(nonzero_idx[1][0:cun_count[0]])]
    for k in range(0, n - 1):
        stratage.append(list(nonzero_idx[1][cun_count[k]:cun_count[k + 1]]))

    ## all to all stratage
    if all2all_flag == True:
        for k in range(1, n):
            temp = []
            len_k = len(stratage[k])
            if k + 1 in stratage[k]:
                idx = stratage[k].index(k + 1)
            else:
                idx = np.argmax(np.array(stratage[k]) > k + 1)
            temp.extend(stratage[k][idx:len_k])
            temp.extend(stratage[k][0:idx])
            stratage[k] = temp

    ## random stratage
    if rand_flag == True:
        for k in range(n):
            A = stratage[k]
            random.shuffle(A)
            stratage[k] = A

    start_time = np.random.rand(n) * 0.1  # random initializing start time
    start_time_copy = start_time.copy()
    start_send = np.arange(n)
    start_recieve = [stratage[k][0] for k in range(n)]
    start_zipped = list(zip(start_time, start_send, start_recieve))
    heapq.heapify(start_zipped)

    ## Initialization
    (min_T, k, l) = heapq.heappop(start_zipped)
    node = (min_T + Time[k, l], k, l)
    linked_node = linked_node(node)
    # linked_node.print_link()


    ## Loop
    for idx in range(cun_count[-1]+n-1):
    # for idx in range(100):
        # for idx in range(1000):
        # print(idx)
        # linked_node.print_link()
        recieve = list(linked_node.recieve_node())
        if (len(start_zipped) > 0) and (start_zipped[0][0] < linked_node.min_time_link()):
            (min_T, k, l) = heapq.heappop(start_zipped)
            if (len(start_zipped) > 0) and (start_zipped[0][2] not in recieve):
                node = (min_T + Time[k, l], k, l)
                linked_node.add_link(node)
        else:
            (min_T, k, l) = linked_node.remove_link()
            stratage[k].remove(l)
            send = linked_node.send_node()
            recieve = list(linked_node.recieve_node())
            if len(start_zipped) > 0:
                temp = set(np.nonzero(min_T >= start_time_copy)[0]) - set(send)  # select the nodes that are not sending and already starting communicating
            else:
                temp = set(np.arange(n)) - set(send)
            temp = sorted(temp - set([i for i in temp if not stratage[i]])) # select the nodes that already have finished communicating
            for i in temp:
                if wait_flag == True:
                    j = stratage[i][0]
                    if j not in recieve:
                        node = (min_T + Time[i, j], i, j)
                        linked_node.add_link(node)
                        recieve.extend([j])
                elif set(stratage[i]) - set(recieve):
                    j = min(set(stratage[i]) - set(recieve))
                    node = (min_T + Time[i, j], i, j)
                    linked_node.add_link(node)
                    recieve.extend([j])
    print("minimum time:", min_T, "ms")
    np.save("./Time_%s.npy" % epoch, min_T)


def task(epoch):

    # 此处读入的文件是按[dst, src]用采样的方式计算的 流量/带宽 估计的时间
    # 同一节点内的带宽为100Gbs，非同一节点内带宽为50Gbs
    file = np.load('./sim_time_2.npz')

    if epoch == 0:
        Time = file["first_route_conn_time"]
        main(Time, epoch, rand_flag=False, wait_flag=True, all2all_flag=False)  # sequential stratage
    elif epoch == 1:
        Time = file["first_route_conn_time"]
        main(Time, epoch, rand_flag=True, wait_flag=True, all2all_flag=False)   # random stratage
    elif epoch == 2:
        Time = file["first_route_conn_time"]
        main(Time, epoch, rand_flag=False, wait_flag=False, all2all_flag=False)   # best stratage
    elif epoch == 3:
        Time = file["first_route_conn_time"]
        main(Time, epoch, rand_flag=False, wait_flag=True, all2all_flag=True)   # all2all stratage
    elif epoch == 4:
        Time = file["second_route_conn_time"]
        main(Time, epoch, rand_flag=False, wait_flag=True, all2all_flag=False)  # sequential stratage
    elif epoch == 5:
        Time = file["second_route_conn_time"]
        main(Time, epoch, rand_flag=True, wait_flag=True, all2all_flag=False)   # random stratage
    elif epoch == 6:
        Time = file["second_route_conn_time"]
        main(Time, epoch, rand_flag=False, wait_flag=False, all2all_flag=False)   # best stratage
    elif epoch == 7:
        Time = file["second_route_conn_time"]
        main(Time, epoch, rand_flag=False, wait_flag=True, all2all_flag=True)   # all2all stratage
    # elif epoch == 8:
    #     Time = file["full_conn_time"]
    #     main(Time, epoch, rand_flag=False, wait_flag=True, all2all_flag=False)  # sequential stratage
    # elif epoch == 9:
    #     Time = file["full_conn_time"]
    #     main(Time, epoch, rand_flag=True, wait_flag=True, all2all_flag=False)   # random stratage
    # elif epoch == 10:
    #     Time = file["full_conn_time"]
    #     main(Time, epoch, rand_flag=False, wait_flag=False, all2all_flag=False)   # best stratage
    # elif epoch == 11:
    #     Time = file["full_conn_time"]
    #     main(Time, epoch, rand_flag=False, wait_flag=True, all2all_flag=True)   # all2all stratage


p = Pool(8)  # Creat multiple process
for epoch in range(8):
    p.apply_async(task, args=(epoch,))
p.close()  # Close pool in case other tasks are submitted into the pool
p.join()  # Waite all the processes in the pool is finished
