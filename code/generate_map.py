"""
体素->dcu映射表的生成过程
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from functions import Functions

'''
generate_my_map()中交换体素的部分写得太丑，可以重构
'''


class GenerateMap(Functions):
    def __init__(self):
        super().__init__()

        # map表地址
        self.map_root = self.root_path + 'tables/map_table/'
        self.map_table_path = self.map_root + self.map_version + '.pkl'
        self.map_table_without_invalid_idx_path = self.map_root + self.map_version + '_without_invalid_idx.pkl'

        '''
        在按编号顺序分配的情况下，每个dcu中包含的体素/功能柱的个数
        N = 2000， n = 22603时，self.voxels_per_dcu = {11: 1397, 12: 603}
        N = 4000， n = 22603时，self.voxels_per_dcu = {5: 1397, 6: 2603}
        N = 17280，n = 171452时，self.voxels_per_dcu = {9: 1348, 10: 15932}
        '''

        # 11*1397+12*603=22603 ,但是前面的进程数的用途？

        self.origin_map_without_invalid_idx = self.generate_sequential_map()  # 按编号顺序将体素分到dcu上
        # self.save_map_pkl(self.origin_map_without_invalid_idx, self.map_table_without_invalid_idx_path)
        # self.origin_map = self.map_table_transfer(self.origin_map_without_invalid_idx)
        # self.save_map_pkl(self.origin_map, self.map_table_path)

    # 生成按编号顺序分组的map表

    # 当N = 2000， n = 22603时，self.voxels_per_dcu = {11: 1397, 12: 603}时
    # 下述循环创建map表，有1397个进程会放置11个体素，603个进程会放置12个体素，并且体素是顺序放置
    def generate_sequential_map(self):
        map_table = list()
        for i in range(self.N):
            map_table.append([])

        cnt, idx = 0, 0
        for key in self.voxels_per_dcu:
            for i in range(self.voxels_per_dcu[key]):
                for j in range(key):
                    map_table[idx].append(cnt)
                    cnt += 1
                idx += 1

        return map_table

    # 生成按自己方案划分的group，按出入流量、体素size均分的原则把体素映射到dcu上
    def generate_my_map(self, iter_times=2000, show_iter_process=False, max_rate=1.15):
        path = self.traffic_table_root + "out_traffic_per_voxel_" + str(self.neuron_number) + ".npy"
        if os.path.exists(path):
            self.out_traffic, self.in_traffic = np.load(path), np.array([])
        else:
            self.out_traffic, self.in_traffic = self.cal_traffic_base_voxel()

        map_table = self.origin_map_without_invalid_idx.copy()

        origin_out, origin_in = self.sum_traffic_per_dcu(self.origin_map_without_invalid_idx)
        origin_size = list()
        for i in range(self.N):
            origin_size.append(np.sum(self.size[np.ix_(self.origin_map_without_invalid_idx[i])]))

        # 画图展示迭代过程，以及与迭代前相比的区别
        if show_iter_process:
            plt.ion()  # 开启interactive mode 成功的关键函数
            plt.figure(figsize=(19, 10))

        time1 = time.time()
        print('Begin to generate map...')
        # for i in range(iter_times):
        sum_out, sum_in = self.sum_traffic_per_dcu(map_table)
        size_per_group = np.array(origin_size)
        cnt = 0

        best_obj = 999

        while np.max(size_per_group) > np.average(size_per_group) * max_rate:
            # print("cnt=%d, average=%f, max/average=%f" %
            #       (cnt, np.average(size_per_group), np.max(size_per_group) / np.average(size_per_group)))
            best_obj = min(best_obj, np.max(size_per_group) / np.average(size_per_group))
            if cnt % 1000 == 0:
                print('best_obj: %.4f, target: %.4f' % (best_obj, max_rate))
            cnt += 1

            # out
            # 在前10大的组中随机挑一个
            # 找出流量最大的dcu与流量最小的dcu
            copy_sum_out = sum_out.copy()
            copy_sum_out.sort()
            idx_1 = np.random.randint(1, 10)
            idx_out_max = np.where(sum_out == copy_sum_out[-idx_1])[0][0]
            idx_out_min = np.argmin(sum_out)

            # 找出流量最大/最小的dcu所包含的功能柱各自的流量
            temp_max = self.out_traffic[np.ix_(map_table[idx_out_max])]
            temp_min = self.out_traffic[np.ix_(map_table[idx_out_min])]

            # 找随机第1-3大的功能柱
            copy_temp_max = temp_max.copy()
            copy_temp_max.sort()
            idx_2 = np.random.randint(1, 3)
            voxel_idx_out_max = np.where(temp_max == copy_temp_max[-idx_2])[0][0]
            voxel_idx_out_min = np.argmin(temp_min)

            max_voxel_overall_idx = map_table[idx_out_max][voxel_idx_out_max]
            min_voxel_overall_idx = map_table[idx_out_min][voxel_idx_out_min]

            temp = map_table[idx_out_max][voxel_idx_out_max]
            map_table[idx_out_max][voxel_idx_out_max] = map_table[idx_out_min][voxel_idx_out_min]
            map_table[idx_out_min][voxel_idx_out_min] = temp

            # 更新流量和
            sum_out[idx_out_max] = sum_out[idx_out_max] + temp_min[voxel_idx_out_min] - temp_max[voxel_idx_out_max]
            sum_out[idx_out_min] = sum_out[idx_out_min] - temp_min[voxel_idx_out_min] + temp_max[voxel_idx_out_max]

            # 更新size
            size_per_group[idx_out_max] = size_per_group[idx_out_max] + self.size[min_voxel_overall_idx] - self.size[
                max_voxel_overall_idx]
            size_per_group[idx_out_min] = size_per_group[idx_out_min] - self.size[min_voxel_overall_idx] + self.size[
                max_voxel_overall_idx]

            # in
            # # 找随机第1-10大的组
            # copy_sum_in = sum_in.copy()
            # copy_sum_in.sort()
            # idx = np.random.randint(1, 10)
            # idx_in_max = np.where(sum_in == copy_sum_in[-idx])[0][0]
            #
            # idx_in_min = np.argmin(sum_in)
            # temp_max = self.in_traffic[np.ix_(map_table[idx_in_max])]
            # temp_min = self.in_traffic[np.ix_(map_table[idx_in_min])]
            #
            # # 找随机第1-4大的体素
            # copy_temp_max = temp_max.copy()
            # copy_temp_max.sort()
            # idx = np.random.randint(1, 2)
            # voxel_idx_in_max = np.where(temp_max == copy_temp_max[-idx])[0][0]
            # voxel_idx_in_min = np.argmin(temp_min)
            #
            # temp = map_table[idx_in_max][voxel_idx_in_max]
            # map_table[idx_in_max][voxel_idx_in_max] = map_table[idx_in_min][voxel_idx_in_min]
            # map_table[idx_in_min][voxel_idx_in_min] = temp

            # size

            # size
            max_size_idx = np.argmax(size_per_group)
            min_size_idx = np.argmin(size_per_group)
            max_size_voxel_idx = np.argmax(self.size[np.ix_(map_table[max_size_idx])])
            min_size_voxel_idx = np.argmin(self.size[np.ix_(map_table[min_size_idx])])

            max_voxel_overall_idx = map_table[max_size_idx][max_size_voxel_idx]
            min_voxel_overall_idx = map_table[min_size_idx][min_size_voxel_idx]

            temp = map_table[max_size_idx][max_size_voxel_idx]
            map_table[max_size_idx][max_size_voxel_idx] = map_table[min_size_idx][min_size_voxel_idx]
            map_table[min_size_idx][min_size_voxel_idx] = temp

            # 更新size
            size_per_group[max_size_idx] = \
                size_per_group[max_size_idx] - self.size[max_voxel_overall_idx] + self.size[min_voxel_overall_idx]
            size_per_group[min_size_idx] = \
                size_per_group[min_size_idx] + self.size[max_voxel_overall_idx] - self.size[min_voxel_overall_idx]

            # 更新流量和
            traffic_max = self.out_traffic[max_voxel_overall_idx]
            traffic_min = self.out_traffic[min_voxel_overall_idx]
            sum_out[max_size_idx] = sum_out[max_size_idx] + traffic_min - traffic_max
            sum_out[min_size_idx] = sum_out[min_size_idx] - traffic_min + traffic_max

            # 画图查看迭代结果
            # if show_iter_process and i % 100 == 0:
            #     plt.clf()  # 清空之前画的
            #
            #     plt.subplot(211)
            #     plt.title('conn_number: average=%d, max=%d' % (np.average(sum_out), np.max(sum_out)))
            #     plt.xlabel('dcu')
            #     plt.ylabel('conn_number')
            #     plt.plot(origin_out, color='blue', linestyle='--', alpha=0.3)
            #     plt.plot(sum_out, color='blue')
            #
            #     plt.subplot(212)
            #     plt.title('size: average=%.6f, max=%.6f' % (np.average(size_per_group), np.max(size_per_group)))
            #     plt.xlabel('dcu')
            #     plt.ylabel('size')
            #     plt.plot(origin_size, linestyle='--', color='green', alpha=0.3)
            #     plt.plot(size_per_group, color='green')
            #
            #     plt.pause(0.05)

            # self.show_progress(i, iter_times, time1)

        time2 = time.time()
        print('Map generated. Iter times: %d, %.2fs consumed' % (iter_times, (time2 - time1)))

        best_obj = min(best_obj, np.max(size_per_group) / np.average(size_per_group))
        print('best_obj: %.4f, target: %.4f' % (best_obj, max_rate))

        # 展示每张dcu向外的流量之和
        sum_out, sum_in = self.sum_traffic_per_dcu(map_table)
        self.draw_traffic(origin_out, sum_out, name="traffic_out")

        # 展示每张dcu包含体素的size之和
        self.draw_traffic(origin_size, size_per_group, name="size")

        if self.map_table_without_invalid_idx_path is not None:
            self.save_map_pkl(map_table, self.map_table_without_invalid_idx_path)

        map_table = self.map_table_transfer(map_table)

        if self.map_table_path is not None:
            self.save_map_pkl(map_table, self.map_table_path)

        self.save_map_pkl(self.origin_map_without_invalid_idx, self.map_root + 'map_1200_sequential.pkl')

        return map_table

    # 把degree*size作为优化指标
    def generate_my_map_new(self, max_rate, show_iter_process=False):
        self.show_basic_information()
        map_table = self.origin_map_without_invalid_idx

        origin_size_degree = self.cal_size_multi_degree(map_table)

        # 画图展示迭代过程，以及与迭代前相比的区别
        if show_iter_process:
            plt.ion()  # 开启interactive mode 成功的关键函数
            plt.figure(figsize=(10, 6), dpi=100)

        time1 = time.time()
        print('Begin to generate map...')
        size_per_dcu = np.array(origin_size_degree)
        cnt = 0

        best_obj = 999

        while np.max(size_per_dcu) > np.average(size_per_dcu) * max_rate:
            # print("cnt=%d, average=%f, max/average=%f" %
            #       (cnt, np.average(size_per_dcu), np.max(size_per_dcu) / np.average(size_per_dcu)))
            best_obj = min(best_obj, np.max(size_per_dcu) / np.average(size_per_dcu))
            if cnt % 5000 == 0:
                print('iter %d: best_obj: %.8f, target: %.4f' % (cnt, best_obj, max_rate))
                # print('average = %.6f' % np.average(size_per_dcu))
            cnt += 1

            # size
            copy_size_per_dcu = size_per_dcu.copy()
            copy_size_per_dcu.sort()
            idx_temp1 = np.random.randint(1, 2)
            idx_temp2 = np.random.randint(1, 60)
            max_size_idx = np.where(size_per_dcu == copy_size_per_dcu[-idx_temp1])[0][0]
            min_size_idx = np.where(size_per_dcu == copy_size_per_dcu[idx_temp2 - 1])[0][0]

            temp1 = self.size_multi_degree[np.ix_(map_table[max_size_idx])].copy()
            temp2 = temp1.copy()
            temp2.sort()
            idx_temp = np.random.randint(1, 4)
            max_size_voxel_idx = np.where(temp1 == temp2[-idx_temp])[0][0]

            temp1 = self.size_multi_degree[np.ix_(map_table[min_size_idx])]
            temp2 = temp1.copy()
            temp2.sort()
            idx_temp = np.random.randint(1, 2)
            min_size_voxel_idx = np.where(temp1 == temp2[idx_temp - 1])[0][0]

            max_voxel_overall_idx = map_table[max_size_idx][max_size_voxel_idx]
            min_voxel_overall_idx = map_table[min_size_idx][min_size_voxel_idx]

            temp = map_table[max_size_idx][max_size_voxel_idx]
            map_table[max_size_idx][max_size_voxel_idx] = map_table[min_size_idx][min_size_voxel_idx]
            map_table[min_size_idx][min_size_voxel_idx] = temp

            # 更新size
            size_per_dcu[max_size_idx] = \
                size_per_dcu[max_size_idx] - self.size[max_voxel_overall_idx] * self.degree[
                    max_voxel_overall_idx] + self.size[min_voxel_overall_idx] * self.degree[min_voxel_overall_idx]
            size_per_dcu[min_size_idx] = \
                size_per_dcu[min_size_idx] + self.size[max_voxel_overall_idx] * self.degree[
                    max_voxel_overall_idx] - self.size[min_voxel_overall_idx] * self.degree[min_voxel_overall_idx]

            if show_iter_process and cnt % 1000 == 0:
                plt.clf()
                plt.title('iter: %d, size: average=%.6f, max / average =%.6f' %
                          (cnt, np.average(size_per_dcu), np.max(size_per_dcu) / np.average(size_per_dcu)))
                plt.xlabel('dcu')
                plt.ylabel('size')
                plt.plot(origin_size_degree, linestyle='--', color='green', alpha=0.3)
                plt.plot(size_per_dcu, color='green')

                plt.pause(0.5)

        time2 = time.time()
        print('Map generated. Iter times: %d, %.2fs consumed' % (cnt, (time2 - time1)))

        best_obj = min(best_obj, np.max(size_per_dcu) / np.average(size_per_dcu))
        print('best_obj: %.6f, target: %.6f' % (best_obj, max_rate))

        ultimate_size_degree = self.cal_size_multi_degree(map_table)
        print('Check Size Degree: %.6f' % (np.max(ultimate_size_degree) / np.average(ultimate_size_degree)))

        # 展示每张dcu包含体素的size之和
        self.draw_traffic(origin_size_degree, size_per_dcu, name="size_degree")

        if self.conn_version[0:5] != 'voxel':
            self.save_map_pkl(map_table, self.map_table_without_invalid_idx_path)

        map_table = self.map_table_transfer(map_table)

        if self.map_table_path is not None:
            self.save_map_pkl(map_table, self.map_table_path)

        # self.save_map_pkl(self.origin_map, self.map_root + self.map_version + '_sequential.pkl')

        return map_table

    # map表转换函数，存在疑问？
    def map_table_transfer(self, map_table):
        mp_171452_to_226030 = list()
        for i in range(len(self.origin_size)):
            if self.origin_size[i] != 0:
                mp_171452_to_226030.append(i)

        for i in range(len(map_table)):
            for j in range(len(map_table[i])):
                map_table[i][j] = mp_171452_to_226030[map_table[i][j]]

        return map_table

    def cal_size_multi_degree(self, map_table):
        size_degree = np.zeros(self.N)
        for i in range(self.N):
            for cortical_idx in map_table[i]:
                size_degree[i] += self.size[cortical_idx] * self.degree[cortical_idx]
        # print(np.max(size_degree) / np.average(size_degree))
        return size_degree

    def draw_traffic(self, traffic_origin, traffic_now, name):
        plt.figure(figsize=(10, 6), dpi=150)
        plt.title(name + ': average=%f, max=%f' % (np.average(traffic_now), np.max(traffic_now)))
        plt.xlabel('dcu')
        plt.ylabel('traffic')

        # plt.xlim(0, self.N)
        # plt.ylim(1000, 4000)

        plt.plot(traffic_origin, color='blue', linestyle='--', alpha=0.3, label='before')
        plt.plot(traffic_now, color='blue', label='now')
        plt.legend(fontsize=13)
        figure_name = name + "_per_dcu_" + self.map_version + ".png"
        plt.savefig(self.root_path + "tables/map_table/" + figure_name)
        print(self.root_path + "tables/map_table/" + figure_name + " saved.")


if __name__ == "__main__":
    g = GenerateMap()
    g.generate_my_map_new(show_iter_process=False, max_rate=1.00247)
