import numpy as np
import time


# import matplotlib.pyplot as plt


class GaForRoute:
    def __init__(self):
        # parameters
        self.max_iter_times = int(100)
        self.cross_rate = 0.4
        self.mutate_rate = 0.2
        self.population_size = 10

        self.N = 2000
        self.n_groups = 50
        self.n_dcus_per_group = 40

        self.route_name = 'route_2000_v4_map_v2.npy'
        self.history_fitness_name = 'history_fitness_map_v3_1200.npy'

        self.history_fitness = list()
        self.history_time = list()

        self.traffic_base_dcu = np.load("../../tables/traffic_table/traffic_table_base_dcu_map_2000_sequential.npy")
        print('traffic table loaded.')

        self.original_out_traffic = self.compute_level2_out_traffic(self.array_to_matrix(np.arange(0, self.N)))
        print('initial: %.4f' % (np.max(self.original_out_traffic) / np.average(self.original_out_traffic)))

        # variables duration evolution
        self.iter_cnt = 0
        self.population = list()

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
        for i in range(self.population_size):
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
                idx = np.random.randint(0, self.population_size)
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

    def select(self):
        fitness_value = np.zeros(len(self.population))
        for i in range(len(self.population)):
            fitness_value[i] = self.fitness_function(self.population[i])
            # print('hey')

        new_population = list()
        fitness_value_after_selection = list()
        sort_idx = np.argsort(fitness_value)
        for i in range(0, self.population_size):
            new_population.append(self.population[sort_idx[i]])
            fitness_value_after_selection.append(fitness_value[sort_idx[i]])

        self.population = new_population.copy()
        print("Best chromosome: %.4f" % np.min(fitness_value_after_selection))
        # print("Group average  : %.4f" % np.average(fitness_value_after_selection))

        # best_level2_out_traffic = self.compute_traffic_per_dcu(self.population[0])
        self.history_fitness.append(np.min(fitness_value_after_selection))
        # print(self.iter_cnt, fitness_value_after_selection[0],
        #       np.max(best_level2_out_traffic) / np.average(best_level2_out_traffic))

        # if self.iter_cnt % 20 == 0:
        #     np.save('serial_best_sizes_' + str(self.iter_cnt) + '.npy', best_level2_out_traffic)
        #     plt.figure(figsize=(9, 5), dpi=200)
        #     plt.title('iter = %d, max / average = %.4f'
        #               % (self.iter_cnt, np.max(best_level2_out_traffic) / np.average(best_level2_out_traffic)))
        #     plt.plot(self.original_out_traffic, color='blue', linestyle='--', alpha=0.3, label='before')
        #     plt.plot(best_level2_out_traffic, color='blue', label='now')
        #     plt.legend(fontsize=15)
        #     plt.show()

    def evolve(self):
        time_start = time.time()
        self.crossover()
        self.mutate()
        self.select()
        self.iter_cnt += 1

        time_end = time.time()
        self.history_time.append(time_end - time_start)
        print('Iter %d: %.2fs consumed.' % (self.iter_cnt, time_end - time_start))
        print('Average time:', sum(self.history_time) / len(self.history_time))

    def begin_evolution(self):
        self.form_initial_population()

        time_start = time.time()
        while self.iter_cnt < self.max_iter_times:
            self.evolve()
        time_end = time.time()
        print('%.4f seconds consumed for %d iteration.' % ((time_end - time_start), self.max_iter_times))

        np.save(self.history_fitness_name, self.history_fitness)
        print(self.history_fitness_name + ' saved.')

        route_table = self.array_to_route(self.population[0])
        np.save(self.route_name, route_table)
        print(self.route_name + ' saved.')

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

    def speed_test(self):
        for population_size in range(10, 110, 10):
            print('Population:', population_size)
            self.population_size = population_size
            self.iter_cnt = 0
            self.begin_evolution()


print('Hello')
if __name__ == '__main__':
    print('Let us go!')
    g = GaForRoute()
    # g.speed_test()
    g.begin_evolution()
