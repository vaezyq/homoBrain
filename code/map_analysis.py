"""
这个类用于分析生成的体素->dcu映射表
"""
import os.path

import numpy as np
import time
import matplotlib.pyplot as plt
from generate_map import GenerateMap


class MapAnalysis(GenerateMap):
    def __init__(self):
        super().__init__()

        self.map_table = self.read_map_pkl(self.map_table_path)

        if self.conn_version[0:5] == 'voxel':
            self.map_table_without_invalid_idx = self.map_table
        else:
            self.map_table_without_invalid_idx = self.read_map_pkl(self.map_table_without_invalid_idx_path)

        self.traffic_voxel_to_voxel = np.array([])

    def cal_probability_per_dcu(self):
        probability_per_dcu = np.zeros(self.N)
        out_size_per_voxel = np.zeros(self.n)

        for i in range(self.n):
            out_size_per_voxel[i] = np.sum(self.conn[:, i])

        for i in range(self.N):
            for voxel_idx in self.map_table[i]:
                probability_per_dcu[i] += out_size_per_voxel[voxel_idx]

        print(np.max(probability_per_dcu), np.average(probability_per_dcu))

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 7), dpi=100)
        plt.title('probability_per_dcu: max = %.4f, average = %.4f, max/average = %.4f'
                  % (np.max(probability_per_dcu), np.average(probability_per_dcu),
                     np.max(probability_per_dcu / np.average(probability_per_dcu))))
        plt.plot(probability_per_dcu)
        plt.show()

        traffic_base_dcu = np.load('traffic_table_base_dcu_map_2000_v2.npy')
        sum_out = np.zeros(self.N)
        sum_in = np.zeros(self.N)

        for i in range(self.N):
            sum_out[i] = np.sum(traffic_base_dcu[:, i])
            sum_in[i] = np.sum(traffic_base_dcu[i, :])

        plt.figure(figsize=(12, 7), dpi=100)
        plt.title('sum_out: max = %.4f, average = %.4f, max/average = %.4f'
                  % (np.max(sum_out), np.average(sum_out), np.max(sum_out / np.average(sum_out))))
        plt.plot(sum_out)
        plt.show()

        plt.figure(figsize=(12, 7), dpi=100)
        plt.title('sum_in: max = %.4f, average = %.4f, max/average = %.4f'
                  % (np.max(sum_in), np.average(sum_in), np.max(sum_in / np.average(sum_in))))
        plt.plot(sum_in)
        plt.show()

        print('hey')

    # 计算划分后，交换机之间的流量
    def cal_flow_under_switch(self):
        map_table = self.map_table

        dcus_per_switch = list()  # 每个交换机下的dcu
        for i in range(self.number_of_switches):
            dcus_per_switch.append([])

        idx = 0
        for i in range(self.number_of_switches):
            for j in range(self.gpu_per_switch):
                dcus_per_switch[i].append(map_table[idx])
                idx += 1

        flow_between_switches = list()  # 交换机两两之间的流量
        for i in range(self.number_of_switches):
            flow_between_switches.append([0] * self.number_of_switches)

        print('Begin calculation...')
        time_start = time.time()
        for i in range(self.number_of_switches):
            for j in range(self.number_of_switches):
                for dcu_out in dcus_per_switch[i]:
                    for voxel_out in dcu_out:
                        for dcu_in in dcus_per_switch[j]:
                            for voxel_in in dcu_in:
                                flow_between_switches[i][j] += self.conn[voxel_out][voxel_in] * self.size[voxel_out]
            # self.show_progress(i, self.number_of_switches, time_start)
        time_end = time.time()
        print('Calculation of the flow between switches complete, %.2fs consumed.' % (time_end - time_start))
        print('hey')

    # 找个数大于1的最小平均值的索引
    @staticmethod
    def find_min_max_idx(initial, average):
        idx_max = int(np.argmax(average))

        idx_all = np.argsort(average)
        temp = 0
        while initial[temp] == 1:
            temp += 1
        idx_min = idx_all[temp]

        return idx_max, idx_min

    # 计算每个dcu平均负责的流量
    def cal_average(self, switch_idx, single_flow, initial):
        average_flow = list()
        for j in range(self.number_of_switches):
            if j < switch_idx:
                average_flow.append(single_flow[j] / initial[j])
            elif j > switch_idx:
                average_flow.append(single_flow[j] / initial[j - 1])
        average_flow = np.array(average_flow)
        return average_flow

    # 根据交换机之间的流量，得出对于每个交换机，向其他交换机发送消息时，要由哪些dcu转发
    def cal_forwarding_table_old(self, iter_times=60):
        flow_between_switches = np.loadtxt('flow_between_switches.txt')  # 交换机两两之间的流量
        forward_table = list()  # 最后输出的dcu负责交换机的情况

        # 初始的负责每个交换机的dcu数量
        initial_num = {3: 16, 4: 8}
        single_forward_table = list()
        for key in initial_num:
            for i in range(initial_num[key]):
                single_forward_table.append(key)

        # 对于每个交换机，迭代若干次以生成初始转发表
        for i in range(self.number_of_switches):
            single_flow = flow_between_switches[i]

            # 迭代若干次，更新initial_num
            initial = single_forward_table.copy()
            average_flow = self.cal_average(i, single_flow, initial)
            for j in range(iter_times):
                idx_max, idx_min = self.find_min_max_idx(initial, average_flow)
                initial[idx_max] += 1
                initial[idx_min] -= 1
                # 更新average_flow
                average_flow = self.cal_average(i, single_flow, initial)

            initial.insert(i, 0)

            ret = list()  # 将dcu的数量转化为编号
            cnt = i * self.gpu_per_switch  # 注意
            for j in range(self.number_of_switches):
                ret.append([])
                for k in range(initial[j]):
                    ret[j].append(cnt)
                    cnt += 1

            forward_table.append(ret)

        return forward_table

    # 每n_dcu_per_group个dcu分为1个group，计算每个group到其他所有dcu的流量
    def cal_flow_between_group_and_dcu(self):
        map_table = self.map_table

        dcus_per_group = list()  # 每个group下包含的dcu的绝对编号
        for i in range(self.number_of_groups):
            dcus_per_group.append([])
            for j in range(self.n_gpu_per_group):
                dcus_per_group[i].append(self.n_gpu_per_group * i + j)

        flow_between_group_and_dcu = list()  # 所有group到其他所有dcu的流量
        for i in range(self.number_of_groups):
            flow_between_group_and_dcu.append([0] * self.N)

        print('Begin calculation...')
        time_start = time.time()
        for group_out_idx in range(self.number_of_groups):
            for dcu_out in dcus_per_group[group_out_idx]:
                for voxel_out in map_table[dcu_out]:
                    for dcu_in in range(self.N):
                        for voxel_in in map_table[dcu_in]:
                            flow_between_group_and_dcu[group_out_idx][dcu_in] += \
                                self.conn[voxel_out][voxel_in] * self.size[voxel_out]

                # self.show_progress(dcu_out, self.N, time_start)
        time_end = time.time()
        # np.savetxt('C:/all/WOW/brain/partition_and_route/flow_table/flow_2000_base_node.txt',
        #            flow_between_group_and_dcu)
        print('Calculation of the flow between group and dcu complete, %.2fs consumed.' % (time_end - time_start))
        print('hey')

    # 根据group之间的流量，得出对于每个group，向其他dcu发送消息时，要由group内的哪些dcu转发
    def generate_forwarding_table(self, iter_times=300, max_link=147):
        flow_table_path = 'C:/all/WOW/brain/partition_and_route/flow_table/flow_2000dcu_40.txt'
        flow_between_group_and_dcu = np.loadtxt(flow_table_path)  # group到其他dcu的流量

        forward_table = list()  # 最后输出的dcu负责交换机的情况
        flow_table = list()

        # 对于每个group，迭代若干次以生成初始转发表
        for i in range(self.number_of_groups):
            single_flow_table = flow_between_group_and_dcu[i]

            single_forward_table = list()  # 生成初始解
            idx = 0
            for j in range(self.n_gpu_per_group):
                single_forward_table.append([])
                for k in range(self.number_of_groups - 1):
                    while idx // self.n_gpu_per_group == i:
                        idx += 1
                    single_forward_table[j].append(idx)
                    idx += 1

            flow_each_dcu = np.zeros(self.n_gpu_per_group)  # 计算流量
            for j in range(self.n_gpu_per_group):
                for dcu_in in single_forward_table[j]:
                    flow_each_dcu[j] += single_flow_table[dcu_in]

            # origin = flow_each_dcu.copy()
            # plt.ion()
            # while np.max(flow_each_dcu) > 0.00025:
            for j in range(iter_times):
                max_idx = int(np.argmax(flow_each_dcu))
                min_idx = int(np.argmin(flow_each_dcu))

                # 从负责转发的流量最大的dcu中，随机选择一个其负责转发的dcu，放入流量最小的dcu的负责转发列表中去
                random_idx = np.random.randint(0, len(single_forward_table[max_idx]))
                dcu_to_move = single_forward_table[max_idx][random_idx]
                single_forward_table[max_idx] = list(np.delete(single_forward_table[max_idx], random_idx))
                single_forward_table[min_idx] = list(np.append(single_forward_table[min_idx], dcu_to_move))
                flow_each_dcu[max_idx] -= single_flow_table[dcu_to_move]
                flow_each_dcu[min_idx] += single_flow_table[dcu_to_move]

                # 如果连接数超过要求，从超过要求的dcu中拿出一个流量小的dcu
                if len(single_forward_table[min_idx]) >= max_link - self.n_gpu_per_group + 2:
                    flows = list()
                    for dcu_in in range(len(single_forward_table[min_idx])):
                        flows.append(single_flow_table[dcu_in])
                    sort_idx = np.argsort(flows)
                    rand_idx = np.random.randint(0, 5)
                    temp = single_forward_table[min_idx][sort_idx[rand_idx]]
                    single_forward_table[min_idx] = list(np.delete(single_forward_table[min_idx], sort_idx[rand_idx]))
                    single_forward_table[max_idx] = list(np.append(single_forward_table[max_idx], temp))
                    flow_each_dcu[min_idx] -= single_flow_table[temp]
                    flow_each_dcu[max_idx] -= single_flow_table[temp]

                # plt.clf()
                # plt.title(j)
                # plt.ylim(0.0003, 0.0005)
                # plt.plot(origin, color='blue', linestyle='--', alpha=0.4)
                # plt.plot(flow_each_dcu, color='blue')
                # plt.pause(0.001)
            #
            # max_len = 0
            # for j in range(len(single_forward_table)):
            #     max_len = max(max_len, len(single_forward_table[j]))
            # print('max len = %d' % max_len)

            forward_table.append(single_forward_table)
            flow_table.append(flow_each_dcu)

        flow_array = list()
        for i in range(self.number_of_groups):
            for j in range(self.n_gpu_per_group):
                flow_array.append(flow_table[i][j])
        # np.savetxt('C:/all/WOW/brain/partition_and_route/route_4000_v1/level2_flow_before.txt', flow_array)

        return forward_table

    def cal_flow_between_group_and_node(self):
        flow_table_path = 'C:/all/WOW/brain/partition_and_route/flow_table/flow_2000dcu_40.txt'
        flow_between_group_and_dcu = np.loadtxt(flow_table_path)  # group到其他dcu的流量

        flow_between_group_and_node = np.zeros((self.number_of_groups, self.number_of_nodes))

        for i in range(self.number_of_groups):
            for j in range(self.number_of_nodes):
                flow_between_group_and_node[i][j] = np.sum(
                    flow_between_group_and_dcu[2 * i: 2 * i + 2, 4 * j: 4 * j + 4])

        np.save('C:/all/WOW/brain/partition_and_route/flow_table/flow_between_25group_and_500node.npy',
                flow_between_group_and_node)

        return

    def generate_forwarding_table_base_node(self, iter_times=30, max_link=51):
        flow_table_path = 'C:/all/WOW/brain/partition_and_route/flow_table/flow_between_25group_and_500node.txt'
        flow_between_group_and_node = np.loadtxt(flow_table_path)  # group到其他node的流量

        forward_table = list()  # 最后输出的dcu负责交换机的情况
        flow_table = list()

        # 对于每个group，迭代若干次以生成初始转发表
        for i in range(self.number_of_groups):
            single_flow_table = flow_between_group_and_node[i]

            # 生成初始解
            single_forward_table = list()
            idx = 0
            for j in range(self.n_node_per_group):
                single_forward_table.append([])
                for k in range(self.number_of_groups - 1):
                    while idx // self.n_node_per_group == i:
                        idx += 1
                    single_forward_table[j].append(idx)
                    idx += 1

            # 计算流量
            flow_each_node = np.zeros(self.n_node_per_group)
            for j in range(self.n_node_per_group):
                for node_in in single_forward_table[j]:
                    flow_each_node[j] += single_flow_table[node_in]

            # origin = flow_each_node.copy()
            # plt.ion()

            # 迭代降低每个node负责转发的流量
            # for j in range(iter_times):  # 指定迭代系数的循环条件
            cnt_iter = 0
            while np.max(flow_each_node) > 0.001837 or cnt_iter < iter_times:  # 限制流量的循环条件
                cnt_iter += 1
                max_idx = int(np.argmax(flow_each_node))
                min_idx = int(np.argmin(flow_each_node))

                # 从负责转发的流量最大的node中，随机选择一个其负责转发的node，放入流量最小的node负责转发的列表中去
                random_idx = np.random.randint(0, len(single_forward_table[max_idx]))
                node_to_move = single_forward_table[max_idx][random_idx]
                single_forward_table[max_idx] = list(np.delete(single_forward_table[max_idx], random_idx))
                single_forward_table[min_idx] = list(np.append(single_forward_table[min_idx], node_to_move))
                flow_each_node[max_idx] -= single_flow_table[node_to_move]
                flow_each_node[min_idx] += single_flow_table[node_to_move]

                # 如果连接数超过要求，从超过要求的node中随机拿出一个流量第0-2小的dcu
                if len(single_forward_table[min_idx]) >= max_link - self.n_node_per_group + 2:
                    flows = list()
                    for node_in in range(len(single_forward_table[min_idx])):
                        flows.append(single_flow_table[node_in])
                    sort_idx = np.argsort(flows)
                    rand_idx = np.random.randint(0, 5)
                    temp = single_forward_table[min_idx][sort_idx[rand_idx]]
                    single_forward_table[min_idx] = list(np.delete(single_forward_table[min_idx], sort_idx[rand_idx]))
                    single_forward_table[max_idx] = list(np.append(single_forward_table[max_idx], temp))
                    flow_each_node[min_idx] -= single_flow_table[temp]
                    flow_each_node[max_idx] -= single_flow_table[temp]

                # plt.clf()
                # # plt.title(j)
                # plt.ylim(0.0010, 0.0016)
                # plt.plot(origin, color='blue', linestyle='--', alpha=0.4)
                # plt.plot(flow_each_node, color='blue')
                # plt.pause(0.001)
            # max_len = 0
            # for j in range(len(single_forward_table)):
            #     max_len = max(max_len, len(single_forward_table[j]))
            # print('max len = %d' % max_len)

            forward_table.append(single_forward_table)
            flow_table.append(flow_each_node)

        # 计算每个node的2级路由流量
        flow_array = list()
        for i in range(self.number_of_groups):
            for j in range(self.n_node_per_group):
                flow_array.append(flow_table[i][j])
        np.save(self.route_path + 'level2_route_traffic.npy', flow_array)

        # 画图看流量变化
        plt.figure(figsize=(10, 6))
        plt.title('max=%.5f, min=%.5f, average=%.5f' %
                  (np.max(flow_array), np.min(flow_array), np.average(flow_array)))
        plt.xlabel('node')
        plt.ylabel('traffic')
        plt.ylim(0.0008, 0.0024)
        plt.plot(flow_array)
        plt.savefig(self.route_path + 'level2_route_traffic.png', dpi=200)
        plt.show()

        return forward_table

    def cal_flow_between_group_and_dcu_unpack_inside_group(self):
        map_table = self.map_table

        dcus_per_group = list()  # 每个group下包含的dcu的绝对编号
        for i in range(self.number_of_groups):
            dcus_per_group.append([])
            for j in range(self.n_gpu_per_group):
                dcus_per_group[i].append(self.number_of_groups * j + i)

        flow_between_group_and_dcu = list()  # 所有group到其他所有dcu的流量
        for i in range(self.number_of_groups):
            flow_between_group_and_dcu.append([0] * self.N)

        print('Begin calculation...')
        time_start = time.time()
        cnt = 0
        for group_out_idx in range(self.number_of_groups):
            for dcu_out in dcus_per_group[group_out_idx]:
                cnt += 1
                for voxel_out in map_table[dcu_out]:
                    for dcu_in in range(self.N):
                        for voxel_in in map_table[dcu_in]:
                            flow_between_group_and_dcu[group_out_idx][dcu_in] += \
                                self.conn[voxel_out][voxel_in] * self.size[voxel_out]

                # self.show_progress(cnt, self.N, time_start)
        end_time = time.time()
        print("Calculation of flow between group and dcu complete. %.2fs consumed." % (end_time - time_start))
        time_end = time.time()
        np.savetxt('C:/all/WOW/brain/partition_and_route/flow_table/flow_2000dcu_40_unpack_inside_group.txt',
                   flow_between_group_and_dcu, dtype=int)
        print('Calculation of the flow between group and dcu complete, %.2fs consumed.' % (time_end - time_start))
        print('hey')

    def generate_forwarding_table_unpack_inside_group(self, iter_times=80, max_link=92):
        flow_table_path = 'C:/all/WOW/brain/partition_and_route/flow_table/flow_2000dcu_40_unpack_inside_group.txt'
        flow_between_group_and_dcu = np.loadtxt(flow_table_path)  # group到其他dcu的流量

        forward_table = list()  # 最后输出的group负责dcu的情况
        flow_table = list()

        # 对于每个group，迭代若干次以生成转发表
        for i in range(self.number_of_groups):
            single_flow_table = flow_between_group_and_dcu[i]

            single_forward_table = list()  # 生成初始解
            idx = 0
            for j in range(self.n_gpu_per_group):
                single_forward_table.append([])
                for k in range(self.number_of_groups):
                    if idx != self.number_of_groups * j + i:
                        single_forward_table[j].append(idx)
                    idx += 1

            flow_each_dcu = np.zeros(self.n_gpu_per_group)  # 计算流量
            for j in range(self.n_gpu_per_group):
                for dcu_in in single_forward_table[j]:
                    flow_each_dcu[j] += single_flow_table[dcu_in]

            # origin = flow_each_dcu.copy()
            # plt.ion()
            # while np.max(flow_each_dcu) > 0.00025:

            for j in range(iter_times):
                max_idx = int(np.argmax(flow_each_dcu))
                min_idx = int(np.argmin(flow_each_dcu))

                # 从负责转发的流量最大的dcu中，随机选择一个其负责转发的dcu，放入流量最小的dcu的负责转发列表中去
                random_idx = np.random.randint(0, len(single_forward_table[max_idx]))
                dcu_to_move = single_forward_table[max_idx][random_idx]
                single_forward_table[max_idx] = list(np.delete(single_forward_table[max_idx], random_idx))
                single_forward_table[min_idx] = list(np.append(single_forward_table[min_idx], dcu_to_move))
                flow_each_dcu[max_idx] -= single_flow_table[dcu_to_move]
                flow_each_dcu[min_idx] += single_flow_table[dcu_to_move]

                # 如果连接数超过要求，从超过要求的dcu中拿出一个流量小的dcu
                if len(single_forward_table[min_idx]) >= max_link - self.n_gpu_per_group + 2:
                    flows = list()
                    for dcu_in in range(len(single_forward_table[min_idx])):
                        flows.append(single_flow_table[dcu_in])
                    sort_idx = np.argsort(flows)
                    rand_idx = np.random.randint(0, 5)
                    temp = single_forward_table[min_idx][sort_idx[rand_idx]]
                    single_forward_table[min_idx] = list(
                        np.delete(single_forward_table[min_idx], sort_idx[rand_idx]))
                    single_forward_table[max_idx] = list(np.append(single_forward_table[max_idx], temp))
                    flow_each_dcu[min_idx] -= single_flow_table[temp]
                    flow_each_dcu[max_idx] -= single_flow_table[temp]

                # plt.clf()
                # plt.title('iter times=%d, max=%.6f, average=%.6f' %
                #           (j, np.max(flow_each_dcu), np.average(flow_each_dcu)))
                # plt.ylim(0.0003, 0.0008)
                # plt.plot(origin, color='blue', linestyle='--', alpha=0.4)
                # plt.plot(flow_each_dcu, color='blue')
                # plt.pause(0.001)

            # max_len = 0
            # for j in range(len(single_forward_table)):
            #     max_len = max(max_len, len(single_forward_table[j]))
            # print('max len = %d' % max_len)

            forward_table.append(single_forward_table)
            flow_table.append(flow_each_dcu)

        return forward_table

    def cal_traffic_between_group_and_dcu_unpack_inside_group_with_sampling(self):
        map_table = self.map_table

        dcus_per_group = list()  # 每个group下包含的dcu的绝对编号
        for i in range(self.number_of_groups):
            dcus_per_group.append([])
            for j in range(self.n_gpu_per_group):
                dcus_per_group[i].append(self.number_of_groups * j + i)

        temp = np.load("../tables/traffic_table/out_traffic_voxel_to_voxel_map_v2.npz")
        self.traffic_voxel_to_voxel = temp[temp.files[1]]

        traffic_between_group_and_dcu = np.zeros((self.number_of_groups, self.N))

        for group_idx in range(self.number_of_groups):
            for dcu_in_idx in range(self.N):
                for voxel_in in map_table[dcu_in_idx]:
                    for dcu_out_idx in dcus_per_group[group_idx]:
                        for voxel_out in map_table[dcu_out_idx]:
                            traffic_between_group_and_dcu[group_idx][dcu_in_idx] += \
                                self.traffic_voxel_to_voxel[voxel_out][voxel_in]
            print(group_idx)

        np.save('../tables/traffic_table/group_to_dcu_map_1200_v3.npy', traffic_between_group_and_dcu)
        print('../tables/traffic_table/group_to_dcu_map_1200_v3.npy saved.')

        return traffic_between_group_and_dcu

    def generate_forwarding_table_unpack_inside_group_with_sampling(self, iter_times=80, max_link=92):
        # flow_table_path = '../tables/traffic_table/group_to_dcu_map_1200_v3.npy'
        # flow_between_group_and_dcu = np.load(flow_table_path)
        flow_between_group_and_dcu = self.cal_traffic_between_group_and_dcu_unpack_inside_group_with_sampling()

        forward_table = list()  # 最后输出的group负责dcu的情况
        flow_table = list()

        # 对于每个group，迭代若干次以生成转发表
        for i in range(self.number_of_groups):
            single_flow_table = flow_between_group_and_dcu[i]

            single_forward_table = list()  # 生成初始解
            idx = 0
            for j in range(self.n_gpu_per_group):
                single_forward_table.append([])
                for k in range(self.number_of_groups):
                    if idx != self.number_of_groups * j + i:
                        single_forward_table[j].append(idx)
                    idx += 1

            flow_each_dcu = np.zeros(self.n_gpu_per_group)  # 计算流量
            for j in range(self.n_gpu_per_group):
                for dcu_in in single_forward_table[j]:
                    flow_each_dcu[j] += single_flow_table[dcu_in]

            # origin = flow_each_dcu.copy()
            # plt.ion()
            # for j in range(iter_times):
            while np.max(flow_each_dcu) > 445000:
                max_idx = int(np.argmax(flow_each_dcu))
                min_idx = int(np.argmin(flow_each_dcu))

                # 从负责转发的流量最大的dcu中，随机选择一个其负责转发的dcu，放入流量最小的dcu的负责转发列表中去
                random_idx = np.random.randint(0, len(single_forward_table[max_idx]))
                dcu_to_move = single_forward_table[max_idx][random_idx]
                single_forward_table[max_idx] = list(np.delete(single_forward_table[max_idx], random_idx))
                single_forward_table[min_idx] = list(np.append(single_forward_table[min_idx], dcu_to_move))
                flow_each_dcu[max_idx] -= single_flow_table[dcu_to_move]
                flow_each_dcu[min_idx] += single_flow_table[dcu_to_move]

                # 如果连接数超过要求，从超过要求的dcu中拿出一个流量小的dcu
                if len(single_forward_table[min_idx]) >= max_link - self.n_gpu_per_group + 2:
                    flows = list()
                    for dcu_in in range(len(single_forward_table[min_idx])):
                        flows.append(single_flow_table[dcu_in])
                    sort_idx = np.argsort(flows)
                    rand_idx = np.random.randint(0, 10)
                    temp = single_forward_table[min_idx][sort_idx[rand_idx]]
                    single_forward_table[min_idx] = list(
                        np.delete(single_forward_table[min_idx], sort_idx[rand_idx]))
                    single_forward_table[max_idx] = list(np.append(single_forward_table[max_idx], temp))
                    flow_each_dcu[min_idx] -= single_flow_table[temp]
                    flow_each_dcu[max_idx] += single_flow_table[temp]

                # plt.clf()
                # plt.title('iter times=%d, max=%.6f, average=%.6f' %
                #           (j, np.max(flow_each_dcu), np.average(flow_each_dcu)))
                # plt.ylim(0.0003, 0.0008)
                # plt.plot(origin, color='blue', linestyle='--', alpha=0.4)
                # plt.plot(flow_each_dcu, color='blue')
                # plt.pause(0.001)

            # max_len = 0
            # for j in range(len(single_forward_table)):
            #     max_len = max(max_len, len(single_forward_table[j]))
            # print('max len = %d' % max_len)

            forward_table.append(single_forward_table)
            flow_table.append(flow_each_dcu)

        flow_array = list()
        for i in range(self.number_of_groups):
            for j in range(self.n_gpu_per_group):
                flow_array.append(flow_table[i][j])

        plt.plot(flow_array)
        plt.title('max=%d, average=%d' % (np.max(flow_array), np.average(flow_array)))
        plt.show()

        np.save(self.route_path + 'forwarding_table.npy', flow_array)

        return forward_table

    def cal_traffic_between_group_and_dcu(self, n_gpu_per_group):
        traffic_table_base_dcu = np.load(
            self.traffic_table_root + 'traffic_table_base_dcu_' + self.map_version + '.npy')

        traffic_group_to_dcu = np.zeros((n_gpu_per_group, self.N))

        start_time = time.time()
        for i in range(self.N):
            for j in range(self.N):
                group_idx = i % n_gpu_per_group
                traffic_group_to_dcu[group_idx][j] += traffic_table_base_dcu[j][i]
            self.show_progress(i, self.N, start_time)

        # plt.figure(figsize=(10, 6), dpi=200)
        # plt.plot(traffic_group_to_dcu)
        print()
        print(np.max(traffic_group_to_dcu), np.min(traffic_group_to_dcu), np.average(traffic_group_to_dcu))

        np.save(self.route_path + 'traffic_group_to_dcu.npy', traffic_group_to_dcu)
        print(self.route_path + 'traffic_group_to_dcu.npy saved.')

    def generate_forwarding_table_17280(self, number_of_group, dcu_per_group, max_link, max_rate):
        assert number_of_group * dcu_per_group == self.N

        if not os.path.exists(self.route_path + 'traffic_group_to_dcu.npy'):
            self.cal_traffic_between_group_and_dcu(self.n_gpu_per_group)

        traffic_group_to_dcu = np.load(self.route_path + 'traffic_group_to_dcu.npy')

        traffic_per_dcu = np.empty(0)
        forwarding_table = list()

        for i in range(dcu_per_group):
            single_traffic_table = traffic_group_to_dcu[i]

            single_forward_table = list()  # 生成初始解
            idx = 0
            for j in range(number_of_group):
                single_forward_table.append([])
                for k in range(dcu_per_group):
                    if idx != dcu_per_group * j + i:
                        single_forward_table[j].append(idx)
                    idx += 1

            # 计算每个dcu转发的流量
            level2_traffic_per_dcu = np.zeros(number_of_group)
            for j in range(number_of_group):
                for idx in single_forward_table[j]:
                    level2_traffic_per_dcu[j] += single_traffic_table[idx]

            # 迭代使转发的流量更加平均
            while np.max(level2_traffic_per_dcu) > np.average(traffic_group_to_dcu) * dcu_per_group * max_rate:
                max_idx = int(np.argmax(level2_traffic_per_dcu))
                min_idx = int(np.argmin(level2_traffic_per_dcu))

                # 从负责转发的流量最大的dcu中，随机选择一个其负责转发的dcu，放入流量最小的dcu的负责转发列表中去
                random_idx = np.random.randint(0, len(single_forward_table[max_idx]))
                dcu_to_move = single_forward_table[max_idx][random_idx]
                single_forward_table[max_idx] = list(np.delete(single_forward_table[max_idx], random_idx))
                single_forward_table[min_idx] = list(np.append(single_forward_table[min_idx], dcu_to_move))
                level2_traffic_per_dcu[max_idx] -= single_traffic_table[dcu_to_move]
                level2_traffic_per_dcu[min_idx] += single_traffic_table[dcu_to_move]

                # 如果连接数超过要求，从超过要求的dcu中拿出一个流量小的dcu
                if len(single_forward_table[min_idx]) >= max_link - number_of_group + 2:
                    flows = list()
                    for dcu_in in range(len(single_forward_table[min_idx])):
                        flows.append(single_traffic_table[dcu_in])
                    sort_idx = np.argsort(flows)
                    rand_idx = np.random.randint(0, 10)
                    temp = single_forward_table[min_idx][sort_idx[rand_idx]]
                    single_forward_table[min_idx] = list(
                        np.delete(single_forward_table[min_idx], sort_idx[rand_idx]))
                    single_forward_table[max_idx] = list(np.append(single_forward_table[max_idx], temp))
                    level2_traffic_per_dcu[min_idx] -= single_traffic_table[temp]
                    level2_traffic_per_dcu[max_idx] += single_traffic_table[temp]

            traffic_per_dcu = np.hstack((traffic_per_dcu, level2_traffic_per_dcu))
            forwarding_table.append(single_forward_table)

            link_numbers = list()
            for temp in single_forward_table:
                link_numbers.append(len(temp))
            print('i: %d, max link number = %d' % (i, max(link_numbers)))
            print(np.max(level2_traffic_per_dcu), np.average(traffic_group_to_dcu) * dcu_per_group)

        print('###############################')
        print('max / average = %.4f' % (np.max(traffic_per_dcu) / np.average(traffic_per_dcu)))
        # plt.figure(figsize=(12, 6), dpi=200)
        # plt.title('max = %.f, average = %.f, max/average = %.2f' %
        #           (np.max(traffic_per_dcu), np.average(traffic_per_dcu),
        #            np.max(traffic_per_dcu) / np.average(traffic_per_dcu)))
        # plt.ylim(30000000, 160000000)
        # plt.plot(traffic_per_dcu, linewidth=0.2)
        # plt.savefig(self.route_path + 'level2_traffic.png')
        # plt.show()

        import pickle
        with open(self.route_path + 'forwarding_table.pickle', 'wb') as f:
            pickle.dump(forwarding_table, f)
        print(self.route_path + 'forwarding_table.pickle saved.')

    # 计算用流量/带宽估算的卡之间的传输时间
    def cal_time_simulation_table(self):
        traffic_table_base_voxel = np.load(self.traffic_table_root + "traffic_table_base_voxel.npy")
        traffic_table_base_dcu = np.zeros((self.N, self.N))

        start_time = time.time()
        for dst in range(self.N):
            for src in range(self.N):
                # 计算两个dcu之间的流量
                traffic_between_dcu = 0
                for voxel_src in self.map_table[src]:
                    conn_number_esti_sum = 0
                    for voxel_dst in self.map_table[dst]:
                        conn_number_esti_sum += traffic_table_base_voxel[voxel_src][voxel_dst]

                    # [dsts, src]的连接数估计
                    conn_number = np.unique(np.random.choice(int(self.neuron_number * self.size[voxel_src]),
                                                             int(conn_number_esti_sum), replace=True)).shape[0]
                    # [dsts, srcs]的连接数估计
                    traffic_between_dcu += conn_number

                traffic_table_base_dcu[dst][src] = traffic_between_dcu
            # self.show_progress(dst, 2000, start_time)
        end_time = time.time()
        print("Calculation of time simulation table complete. %.2fs consumed." % (end_time - start_time))

        np.save(self.traffic_table_root + 'time_simulation.npy', traffic_table_base_dcu)

    # 计算映射后每张dcu中体素size之和
    def show_size_per_dcu(self):
        size_origin = list()
        size_now = list()
        for i in range(self.N):
            size_origin.append(np.sum(self.size[np.ix_(self.origin_map_without_invalid_idx[i])]))
            size_now.append(np.sum(self.size[np.ix_(self.map_table_without_invalid_idx[i])]))

        np.save('size_origin.npy', size_origin)
        np.save('size_now.npy', size_now)

        plt.figure(figsize=(10, 6), dpi=100)
        # plt.title('max=%.6f, min=%.6f, average=%.6f' % (np.max(size_now), np.min(size_now), np.average(size_now)))
        plt.xlabel('the number of GPUs')
        plt.ylabel('voxel size')
        # plt.ylim(0, 0.0005)
        plt.plot(size_origin, color='blue', linestyle='--', alpha=0.2, label='sum of voxel size\nbefore partitioning')
        plt.plot(size_now, color='blue', alpha=0.9, label='sum of voxel size\nafter partitioning')
        plt.legend(fontsize=13)
        # plt.savefig('size_before_and_after_map.png')
        plt.show()

        print('hey')

    def show_size_degree_new(self):
        origin_size_degree = np.zeros(self.N)
        size_degree = np.zeros(self.N)

        for gpu_idx in range(self.N):
            for cortical_idx in self.origin_map_without_invalid_idx[gpu_idx]:
                origin_size_degree[gpu_idx] += self.size_multi_degree[cortical_idx]

        for gpu_idx in range(self.N):
            for cortical_idx in self.map_table_without_invalid_idx[gpu_idx]:
                size_degree[gpu_idx] += self.size_multi_degree[cortical_idx]

        plt.figure(figsize=(10, 6), dpi=200)
        plt.title(
            self.map_version + ", size*degree max / average = %.6f" % (np.max(size_degree) / np.average(size_degree)))
        plt.plot(origin_size_degree, color='blue', label="before")
        plt.plot(size_degree, color='red', label="after")
        plt.legend(fontsize=15)
        plt.show()

    # 计算以dcu为单位的连接概率矩阵
    def cal_connection_table_base_dcu(self):
        connection_table_base_dcu = np.zeros((self.N, self.N))

        start_time = time.time()
        for dst in range(self.N):
            for src in range(self.N):
                connection_table_base_dcu[dst][src] = np.sum(self.conn[np.ix_(self.map_table[dst],
                                                                              self.map_table[src])])
            self.show_progress(dst, self.N, start_time)

        print("Nonzero Rate: %.2f" % (np.count_nonzero(connection_table_base_dcu) / (self.N ** 2 / 100)) + "%")
        np.save('connection_table_base_dcu_17280.npy', connection_table_base_dcu)

    # 计算dcu之间的连接矩阵
    def cal_binary_connection_table_base_dcu(self):
        file_name = '../tables/map_table/binary_connection_table_base_dcu_' + self.map_version + '.npy'
        binary_connection_table_base_dcu = np.zeros((self.N, self.N), dtype=bool)
        start_time = time.time()

        print('Begin Calculation...')
        for src in range(self.N):
            for dst in range(self.N):
                for cortical_out in self.map_table[src]:
                    temp = False
                    for cortical_in in self.map_table[dst]:
                        if self.conn[cortical_out][cortical_in] != 0:
                            binary_connection_table_base_dcu[src][dst] = 1
                            temp = True
                            break
                    if temp:
                        break

            self.show_progress(src, self.N, start_time)

        print("Nonzero Rate: %.2f" % (np.count_nonzero(binary_connection_table_base_dcu) / (self.N ** 2 / 100)) + "%")
        np.save(file_name, binary_connection_table_base_dcu)

    def show_out_in_traffic_per_dcu(self):
        self.N = 10000
        traffic_table_base_dcu = np.load(self.traffic_table_root + "traffic_table_base_dcu_" +
                                         self.map_version + ".npy")

        out_traffic_per_dcu = np.zeros(self.N)
        in_traffic_per_dcu = np.zeros(self.N)

        for i in range(self.N):
            out_traffic_per_dcu[i] = np.sum(traffic_table_base_dcu[:, i]) - traffic_table_base_dcu[i][i]
            in_traffic_per_dcu[i] = np.sum(traffic_table_base_dcu[i, :]) - traffic_table_base_dcu[i][i]
            # out_traffic_per_dcu[i] = np.sum(traffic_table_base_dcu[:, i])
            # in_traffic_per_dcu[i] = np.sum(traffic_table_base_dcu[i, :])

        X = np.arange(0, self.N)
        Y = np.full(self.N, np.average(out_traffic_per_dcu))

        plt.figure(figsize=(9, 6), dpi=200)
        plt.title('out: max = %d, average = %d, average = %.4f'
                  % (np.max(out_traffic_per_dcu), np.average(out_traffic_per_dcu),
                     np.max(out_traffic_per_dcu) / np.average(out_traffic_per_dcu)))
        plt.plot(X, out_traffic_per_dcu)
        plt.plot(X, Y, linewidth=3, label="average")
        plt.plot(X, 3 * Y, color="green", linewidth=3, label="3x average")
        # plt.plot(X, 4 * Y, color="red", linewidth=3, label="4x average")
        plt.legend(fontsize=15)
        plt.show()

        plt.figure(figsize=(9, 6), dpi=200)
        plt.title('in: max = %d, average = %d, average = %.4f'
                  % (np.max(in_traffic_per_dcu), np.average(in_traffic_per_dcu),
                     np.max(in_traffic_per_dcu) / np.average(in_traffic_per_dcu)))
        plt.plot(in_traffic_per_dcu)
        plt.plot(X, Y, linewidth=3, label="average")
        plt.legend(fontsize=15)
        plt.show()

        # np.save('map_sequential_out_traffic.npy', out_traffic_per_dcu)
        # np.save('map_sequential_in_traffic.npy', in_traffic_per_dcu)


if __name__ == "__main__":
    m = MapAnalysis()
    m.generate_forwarding_table_17280(number_of_group=100, dcu_per_group=100, max_link=205, max_rate=1.085)
    # m.show_size_degree_new()
    # m.show_out_in_traffic_per_dcu()
    pass
