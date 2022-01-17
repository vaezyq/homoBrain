"""
The class "GenerateMapParallel" generates map table in a parallel way.
When N = 16000, comm_size = 1600:
- It takes about 10 seconds for one iteration
"""

import numpy as np
import time
from generate_map_parallel_master import GenerateMapParallelMaster
from generate_map_parallel_slave import GenerateMapParallelSlave


class GenerateMapParallel(GenerateMapParallelMaster, GenerateMapParallelSlave):
    def __init__(self):
        super().__init__()

        self.map_table = self.read_map_pkl(self.map_table_without_invalid_idx_path)
        self.new_map_version = 'map_' + str(self.N) + '_v4_' + self.conn_version

        self.traffic_base_dcu_saving_path = 'traffic_table_base_dcu_' + self.new_map_version + '.npy'
        self.map_table_saving_path = self.new_map_version + '.pkl'
        self.map_table_without_invalid_idx_saving_path = self.new_map_version + '_without_invalid_idx.pkl'

        self.target = 5
        self.max_iter_count = 100

        self.show_traffic_interval = 1

        self.iter_count = 0
        self.best_map = {'map_table': list(), 'out': 1e9, 'in': 1e9}

    @staticmethod
    def obj_function(map_table_dict):
        # return map_table_dict['out'] + map_table_dict['in']
        return map_table_dict['out']

    def iteration_stop_condition(self):
        # compute the current map's out traffic max / average
        max_traffic_out = np.max(self.traffic_out_per_dcu)
        average_traffic_out = np.average(self.traffic_out_per_dcu)
        object_out = max_traffic_out / average_traffic_out

        # compute the current map's in traffic max / average
        max_traffic_in = np.max(self.traffic_in_per_dcu)
        average_traffic_in = np.average(self.traffic_in_per_dcu)
        object_in = max_traffic_in / average_traffic_in

        # if self.obj_function(self.best_map) > object_out + object_in:
        if self.obj_function(self.best_map) > object_out:
            self.best_map = {'map_table': self.map_table, 'out': object_out, 'in': object_in}

        print("####### iter %d: " % self.iter_count)
        print("out: max / average = %.4f, best = %.4f" % (object_out, self.best_map['out']))
        print("in:  max / average = %.4f, best = %.4f" % (object_in, self.best_map['in']))
        print('top 4 out: ', self.traffic_out_per_dcu[np.ix_(np.argsort(self.traffic_out_per_dcu[-4:]))] / np.average(self.traffic_out_per_dcu))

        stop_condition = self.obj_function(self.best_map) > self.target and self.iter_count < self.max_iter_count

        return stop_condition

    def master_rank_print(self, string_to_print):
        if self.rank == self.master_rank:
            print(string_to_print)

    def save_traffic_during_iteration(self):
        out_filename = self.figure_save_path + "traffic_out" + str(self.iter_count) + ".npy"
        in_file_name = self.figure_save_path + "traffic_in" + str(self.iter_count) + ".npy"

        np.save(out_filename, self.traffic_out_per_dcu)
        np.save(in_file_name, self.traffic_in_per_dcu)

    def update_in_traffic(self):
        cortical_1, cortical_2 = None, None
        if self.rank == self.master_rank:
            cortical_1 = self.find_cortical_with_large_traffic_in()
            cortical_2 = self.find_cortical_with_small_traffic_in()
        cortical_1 = self.comm.bcast(cortical_1, root=self.master_rank)
        cortical_2 = self.comm.bcast(cortical_2, root=self.master_rank)

        self.update_map_table_in(cortical_1, cortical_2)

        if self.rank == self.master_rank:
            self.update_traffic_master(cortical_1, cortical_2)
            self.cal_traffic()  # about 2 seconds
            if self.iter_count % self.show_traffic_interval == 0:
                self.save_traffic_during_iteration()
        else:
            self.update_traffic_slave(cortical_1, cortical_2)

        self.comm.barrier()

    def update_out_traffic(self):
        cortical_1, cortical_2 = None, None
        if self.rank == self.master_rank:
            cortical_1 = self.find_cortical_with_large_traffic_out()
            cortical_2 = self.find_cortical_with_small_traffic_out()
        cortical_1 = self.comm.bcast(cortical_1, root=self.master_rank)
        cortical_2 = self.comm.bcast(cortical_2, root=self.master_rank)

        self.update_map_table_out(cortical_1, cortical_2)

        if self.rank == self.master_rank:
            self.update_traffic_master(cortical_1, cortical_2)
            self.cal_traffic()  # about 2 seconds
            if self.iter_count % self.show_traffic_interval == 0:
                self.save_traffic_during_iteration()
        else:
            self.update_traffic_slave(cortical_1, cortical_2)

        self.comm.barrier()

    def update_size_degree(self):
        pass

    def iterate(self):
        self.update_out_traffic()
        self.update_in_traffic()
        # self.update_size_degree()

    def generate_map_parallel(self):
        time_iteration_start = time.time()

        loop_mark = None
        if self.rank == self.master_rank:
            loop_mark = self.iteration_stop_condition()
        loop_mark = self.comm.bcast(loop_mark, root=self.master_rank)

        while loop_mark:
            time1 = time.time()
            self.iter_count += 1

            self.iterate()

            if self.rank == self.master_rank:
                loop_mark = self.iteration_stop_condition()
            loop_mark = self.comm.bcast(loop_mark, root=self.master_rank)

            time2 = time.time()
            if self.rank == self.master_rank:
                print("%.4f consumed." % (time2 - time1))

        time_iteration_end = time.time()

        # plot and save the result
        if self.rank == self.master_rank:
            print('Map generated. %.2fs consumed.' % (time_iteration_end - time_iteration_start))

            np.save(self.traffic_base_dcu_saving_path, self.traffic_base_dcu)
            print(self.traffic_base_dcu_saving_path + " saved. ")

            if self.map_table_without_invalid_idx_saving_path is not None:
                self.save_map_pkl(self.best_map['map_table'], self.map_table_without_invalid_idx_saving_path)

            print('out: %.4f, in: %.4f' % (self.best_map['out'], self.best_map['in']))

            self.map_table = self.map_table_transfer(self.map_table)

            if self.map_table_path is not None:
                self.save_map_pkl(self.map_table, self.map_table_saving_path)


if __name__ == '__main__':
    Job = GenerateMapParallel()
    Job.generate_map_parallel()
