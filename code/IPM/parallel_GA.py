"""
A parallel genetic algorithm with "master-slave, island" hybrid parallel model.

last modified: lyh 2021.10.17
"""

from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt


class ParallelGeneticAlgorithm:
    def __init__(self):
        self.max_iter_times = int(100)
        self.cross_rate = 0.4
        self.mutate_rate = 0.2
        self.population_size_per_island = 20
        self.core_per_processor = 32

        self.N = 1200
        self.n_groups = 30
        self.n_dcus_per_group = 40
        assert self.n_groups * self.n_dcus_per_group == self.N

        self.history_fitness = list()

        self.traffic_base_dcu = np.load("../../tables/traffic_table/traffic_table_base_dcu_map_1200_v3.npy")

        # initialize MPI
        self.comm = MPI.COMM_WORLD
        self.comm_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.number_of_islands = self.comm_size // self.core_per_processor

        self.master_rank = self.rank // self.core_per_processor * self.core_per_processor
        self.slave_rank = self.rank - self.master_rank
        self.island_idx = self.rank // 32
        # print('Rank %d: master_rank = %d, slave_rank = %d' % (self.rank, self.master_rank, self.slave_rank))

        self.original_out_traffic = self.compute_level2_out_traffic(self.array_to_matrix(np.arange(0, self.N)))
        if self.rank == 0:
            print('Initial fitness: %.4f' % (np.max(self.original_out_traffic) / np.average(self.original_out_traffic)))

        # variables duration evolution
        self.iter_cnt = 0
        self.population = list()

    def array_to_route(self, single_chromosome):
        chromosome_matrix = self.array_to_matrix(single_chromosome)

        # 每个dcu的组号
        group_idx_per_dcu = dict()
        for i in range(self.N):
            dcu_idx = single_chromosome[i]
            group_idx_per_dcu[dcu_idx] = {'group_idx': (i // 40), 'inside_group_idx': (i % 40)}

        route_table = np.zeros((self.N, self.N), dtype=int)

        for src in range(self.N):
            for dst in range(self.N):
                if group_idx_per_dcu[src]['inside_group_idx'] == group_idx_per_dcu[dst]['inside_group_idx'] \
                        or group_idx_per_dcu[src]['group_idx'] == group_idx_per_dcu[dst]['group_idx']:
                    route_table[src][dst] = src
                else:
                    group_idx = group_idx_per_dcu[dst]['group_idx']
                    inside_group_idx = group_idx_per_dcu[src]['inside_group_idx']
                    bridge_node = chromosome_matrix[group_idx][inside_group_idx]
                    route_table[src][dst] = bridge_node

        return route_table

    def array_to_matrix(self, chromosome_array):
        chromosome_matrix = np.empty((self.n_groups, self.n_dcus_per_group), dtype=int)
        for i in range(self.n_groups):
            chromosome_matrix[i] = chromosome_array[self.n_dcus_per_group * i: self.n_dcus_per_group * (i + 1)]

        return chromosome_matrix

    def matrix_to_array(self, chromosome_matrix):
        chromosome_array = np.empty(self.N, dtype=int)
        for i in range(self.n_dcus_per_group):
            chromosome_array[self.n_dcus_per_group * i: self.n_dcus_per_group * (i + 1)] = chromosome_matrix[i]

        return chromosome_array

    # {0} -> {1,2,3,...,15}
    def compute_level1_out_traffic(self):
        level1_out_traffic = np.zeros(self.N)
        for i in range(self.N):
            level1_out_traffic[i] = np.sum(self.traffic_base_dcu[:, i]) - self.traffic_base_dcu[i][i]

        return level1_out_traffic

    # {4,8,12} -> {0,1,2,3}
    # {1,2,3} -> 0
    def compute_level1_in_traffic(self, chromosome_matrix, level2_out_traffic):
        level1_in_traffic = level2_out_traffic.copy()

        for i in range(self.n_groups):
            for j in range(self.n_dcus_per_group):
                destination_idx = chromosome_matrix[i][j]

                # {4,8,12} -> {0}
                for p in range(self.n_groups):
                    if p != i:
                        src = chromosome_matrix[p][j]
                        dst = destination_idx
                        level1_in_traffic[i] += self.traffic_base_dcu[dst][src]

                # {1,2,3} -> {0}
                for q in range(self.n_dcus_per_group):
                    if q != j:
                        src = chromosome_matrix[i][q]
                        dst = destination_idx
                        level1_in_traffic[i] += self.traffic_base_dcu[dst][src]

        return level1_in_traffic

    # {4,8,12} -> {1,2,3}
    def compute_level2_out_traffic(self, chromosome_matrix):
        level2_out_traffic = np.zeros(self.N)

        for i in range(self.n_groups):
            for j in range(self.n_dcus_per_group):
                source_idx = chromosome_matrix[i][j]
                # print("bridge: ", source_idx)
                for p in range(self.n_groups):
                    for q in range(self.n_dcus_per_group):
                        if p != i and q != j:
                            src = chromosome_matrix[p][j]
                            dst = chromosome_matrix[i][q]
                            level2_out_traffic[source_idx] += self.traffic_base_dcu[dst][src]
                            # print(str(src) + "->" + str(dst))
                # print()

        return level2_out_traffic

    # {5,6,7,9,10,11,13,14,15} -> {0}
    def compute_level2_in_traffic(self, chromosome_matrix):
        level2_in_traffic = np.zeros(self.N)

        for i in range(self.n_groups):
            for j in range(self.n_dcus_per_group):
                destination_idx = chromosome_matrix[i][j]
                print("dst: ")
                for p in range(self.n_groups):
                    for q in range(self.n_dcus_per_group):
                        source_idx = chromosome_matrix[p][q]
                        if p != i and q != j:
                            level2_in_traffic[destination_idx] += self.traffic_base_dcu[destination_idx][source_idx]
                            print(source_idx, end=" ")
                print()

        return level2_in_traffic

    def compute_traffic_per_dcu(self, single_chromosome):
        time1 = time.time()
        chromosome_array = single_chromosome.copy()
        chromosome_matrix = self.array_to_matrix(chromosome_array)

        level2_out_traffic = self.compute_level2_out_traffic(chromosome_matrix)
        # level2_in_traffic = self.compute_level2_in_traffic(chromosome_matrix)
        # level1_out_traffic = self.compute_level1_out_traffic()
        # level1_in_traffic = self.compute_level1_in_traffic(chromosome_matrix, level2_out_traffic)

        time2 = time.time()
        # print('%.2f consumed.' % (time2 - time1))

        return level2_out_traffic

    def fitness_function(self, single_chromosome):
        level2_out_traffic = self.compute_traffic_per_dcu(single_chromosome)

        return np.max(level2_out_traffic) / np.average(level2_out_traffic)

    def form_initial_population(self):
        single_population = np.arange(0, self.N)
        for i in range(self.population_size_per_island):
            self.population.append(single_population.copy())
            np.random.shuffle(single_population)

    def correct_conflict(self, chromosome, hash_table, loc1, loc2):
        for i in range(self.N):
            if i < loc1 or i > loc2:
                value = chromosome[i]
                while value in hash_table:
                    value = hash_table[value]
                chromosome[i] = value

        return chromosome

    def crossover(self):
        n = len(self.population)
        for i in range(n):
            if np.random.random() < self.cross_rate:
                idx = np.random.randint(0, self.population_size_per_island)
                chromosome1 = self.population[i]
                chromosome2 = self.population[idx]

                loc1 = np.random.randint(0, self.N)
                loc2 = np.random.randint(loc1, min(self.N, loc1 + 80))

                new_chromosome1 = chromosome1.copy()
                new_chromosome2 = chromosome2.copy()

                new_chromosome1[loc1:loc2 + 1] = chromosome2[loc1:loc2 + 1]
                new_chromosome2[loc1:loc2 + 1] = chromosome1[loc1:loc2 + 1]

                hash_table1, hash_table2 = dict(), dict()
                for j in range(loc1, loc2 + 1):
                    hash_table1[chromosome2[j]] = chromosome1[j]
                    hash_table2[chromosome1[j]] = chromosome2[j]

                # print("Before crossover:")
                # print(chromosome1)
                # print(chromosome2)
                # print("Loc1", loc1, "Loc2", loc2)
                # print("Before correction:")
                # print(new_chromosome1)
                # print(new_chromosome2)

                new_chromosome1 = self.correct_conflict(new_chromosome1, hash_table1, loc1, loc2)
                new_chromosome2 = self.correct_conflict(new_chromosome2, hash_table2, loc1, loc2)

                self.population.append(new_chromosome1.copy())
                self.population.append(new_chromosome2.copy())

                # print("After correction:")
                # print(new_chromosome1)
                # print(new_chromosome2)
                #
                # print('hey')

    def mutate(self):
        n = len(self.population)
        for i in range(n):
            if np.random.random() < self.cross_rate:
                chromosome = self.population[i].copy()
                loc1 = np.random.randint(0, self.N)
                loc2 = np.random.randint(loc1, min(self.N, loc1 + 80))
                # print('Before:', chromosome)
                # print('Loc1:', loc1, 'Loc2:', loc2)
                if loc1 == 0:
                    chromosome[loc1:loc2 + 1] = chromosome[loc2::-1]
                else:
                    chromosome[loc1:loc2 + 1] = chromosome[loc2:loc1 - 1:-1]
                # print('After: ', chromosome)
                self.population.append(chromosome)

    def bcast_population_size_inside_island(self):
        if self.rank % self.core_per_processor == 0:
            self.crossover()
            # print('crossover complete.')
            self.mutate()
            # print('mutate complete.')

            # 把计算任务分给在同一个节点内部的其他31个核心
            population_size = len(self.population)
            # print('population size:', population_size)
            for i in range(1, self.core_per_processor):
                self.comm.send(population_size, dest=self.master_rank + i, tag=0)
        else:
            # print('Rank ', self.rank, 'master_rank =', master_rank, 'slave rank =', slave_rank)
            population_size = self.comm.recv(source=self.master_rank, tag=0)
            # print('slave rank %d: %d' % (self.rank, population_size))

        return population_size

    def send_recv_population(self, population_size):
        chromosomes = list()
        if self.rank % self.core_per_processor == 0:
            for i in range(population_size):
                destination_rank = self.rank + (i % (self.core_per_processor - 1)) + 1
                self.comm.send(self.population[i], dest=destination_rank, tag=1)
                # print('population %d sent' % i)
        else:
            if self.slave_rank <= population_size % (self.core_per_processor - 1):
                cnt_population = population_size // (self.core_per_processor - 1) + 1
            else:
                cnt_population = population_size // (self.core_per_processor - 1)

            # print('slave_rank %d, cnt_population %d' % (self.slave_rank, cnt_population))

            for i in range(cnt_population):
                msg = self.comm.recv(source=self.master_rank, tag=1)
                # print('rank %d received a chromosome from rank %d' % (self.rank, self.master_rank))
                chromosomes.append(msg)

        return chromosomes

    def compute_and_recv_fitness_in_parallel(self, population_size, chromosomes):
        if self.rank % self.core_per_processor == 0:
            # receive fitness values computed by slave ranks
            fitness_values = np.zeros(len(self.population))
            for i in range(population_size):
                source_rank = self.rank + i % (self.core_per_processor - 1) + 1
                # print('source rank:', source_rank)
                msg = self.comm.recv(source=source_rank, tag=2)
                # print('fitness %d received.' % i)
                fitness_values[i] = msg

            new_population = list()
            fitness_value_after_selection = list()
            sort_idx = np.argsort(fitness_values)
            for i in range(0, self.population_size_per_island):
                new_population.append(self.population[sort_idx[i]])
                fitness_value_after_selection.append(fitness_values[sort_idx[i]])

            self.population = new_population.copy()
            print("Best chromosome: %.4f from island %d" % (np.min(fitness_value_after_selection),
                                                            self.rank // self.core_per_processor))
            # print("Group average  : %.4f" % np.average(fitness_value_after_selection))

            self.history_fitness.append(np.min(fitness_value_after_selection))
            # best_level2_out_traffic = self.compute_traffic_per_dcu(self.population[0])
            # print(self.iter_cnt, fitness_value_after_selection[0],
            #       np.max(best_level2_out_traffic) / np.average(best_level2_out_traffic))
        else:
            if self.slave_rank <= population_size % (self.core_per_processor - 1):
                cnt_population = population_size // (self.core_per_processor - 1) + 1
            else:
                cnt_population = population_size // (self.core_per_processor - 1)

            # print('slave_rank %d, cnt_population %d' % (self.slave_rank, cnt_population))

            for i in range(cnt_population):
                single_chromosome = chromosomes[i]
                fitness = self.fitness_function(single_chromosome)
                self.comm.send(fitness, dest=self.master_rank, tag=2)
                # print('rank %d sent a fitness' % self.rank)

    def evolve(self):
        self.comm.barrier()
        time_start = time.time()

        population_size = self.bcast_population_size_inside_island()
        # print('####### bcast population size complete #######')

        chromosomes = self.send_recv_population(population_size)
        # print('####### send and recv complete #######')

        self.compute_and_recv_fitness_in_parallel(population_size, chromosomes)
        # print('####### selection complete #######')

        self.iter_cnt += 1

        self.comm.barrier()
        time_end = time.time()
        if self.rank == 0:
            print('Iter %d: %.2f consumed.' % (self.iter_cnt, time_end - time_start))

    def reduce_and_save_the_best_solution(self):
        # reduce to get the best solution
        if self.rank == 0:
            population, fitness = list(), list()
            for source_rank in range(self.core_per_processor, self.comm_size, self.core_per_processor):
                msg = self.comm.recv(source=source_rank)
                population.append(msg['population'])
                fitness.append(msg['fitness'])

            fitness = np.array(fitness)
            min_fitness_idx = np.where(fitness == np.min(fitness))[0][0]
            final_route_array = population[min_fitness_idx]

            print('Final fitness: ', fitness[min_fitness_idx], self.fitness_function(final_route_array))

            route_table = self.array_to_route(final_route_array)
            np.save('route_1200_v2_map_v3.npy', route_table)
            print('route_1200_v2_map_v3.npy saved.')
        elif self.rank % self.core_per_processor == 0:
            msg = {'population': self.population[0], 'fitness': self.fitness_function(self.population[0])}
            self.comm.send(msg, dest=0)

        # if self.rank % self.core_per_processor == 0:
        #     np.save('history_fitness_' + str(self.island_idx) + '.npy')

    def begin_evaluation(self):

        self.form_initial_population()

        time_start = time.time()
        while self.iter_cnt < self.max_iter_times:
            self.evolve()
        time_end = time.time()
        if self.rank == 0:
            print('%d nodes used.' % (self.comm_size // 32))
            print('Iteration %d, %.4f consumed in total' % (self.max_iter_times, (time_end - time_start)))

        self.reduce_and_save_the_best_solution()

    # 测试1个node在不同population size下的加速效果(NSGA-II Fig 8)
    def speed_test(self):
        result = list()
        for population_size in range(10, 110, 10):

            time_start = time.time()
            self.population_size_per_island = population_size
            self.iter_cnt = 0
            self.begin_evaluation()
            time_end = time.time()

            one_result = {'processors': self.comm_size // self.core_per_processor,
                          'population_size': self.population_size_per_island,
                          'iteration_times': self.max_iter_times,
                          'time_consumed': time_end - time_start}
            result.append(one_result)
            if self.rank == 0:
                print('#######################')
                for key in one_result:
                    print(key, one_result[key])
                print('#######################')

        if self.rank == 0:
            time_consumed = np.zeros(10, dtype=int)
            cnt = 0
            for element in result:
                print('#######################')
                for key in element:
                    print(key, element[key])
                time_consumed[cnt] = element['time_consumed']

            np.save('time_consumed_for_diff_population_size.npy', time_consumed)

    # 测试同一population size，不同处理器数量下的加速效果(NSGA-II Fig 10)
    def test_speed_2(self):
        pass


if __name__ == "__main__":
    job = ParallelGeneticAlgorithm()
    job.begin_evaluation()
    pass
