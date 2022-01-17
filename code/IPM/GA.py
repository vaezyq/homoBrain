import numpy as np
import time
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self):
        # parameters
        self.max_iter_times = int(2e4)
        self.cross_rate = 0.1
        self.mutate_rate = 0.1
        self.population_size = 100

        self.N = 2000
        self.n = 22603
        self.voxels_per_dcu = {11: 1397, 12: 603}

        self.size = np.load("../../tables/conn_table/voxel/size.npy")

        self.origin_size_per_dcu = self.compute_size_per_dcu(np.arange(0, self.n))
        self.average_size = np.sum(self.size) / self.N
        print("Average size:", self.average_size)

        # variables duration evolution
        self.iter_cnt = 0
        self.population = list()

    def compute_size_per_dcu(self, single_chromosome):
        sizes = np.zeros(self.N)

        voxel_loc, gpu_idx = 0, 0
        for key in self.voxels_per_dcu:
            for i in range(self.voxels_per_dcu[key]):
                for j in range(key):
                    voxel_idx = single_chromosome[voxel_loc]
                    sizes[gpu_idx] += self.size[voxel_idx]
                    voxel_loc += 1
                gpu_idx += 1

        return sizes

    def fitness_function(self, single_chromosome):
        sizes = self.compute_size_per_dcu(single_chromosome)

        return np.max(sizes) / self.average_size

    def form_initial_population(self):
        single_population = np.arange(0, self.n)
        for i in range(self.population_size):
            self.population.append(single_population.copy())
            np.random.shuffle(single_population)

    def correct_conflict(self, chromosome, hash_table, loc1, loc2):
        for i in range(self.n):
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

                loc1 = np.random.randint(0, self.n)
                loc2 = np.random.randint(loc1, self.n)

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

    def mutate(self):
        n = len(self.population)
        for i in range(n):
            if np.random.random() < self.cross_rate:
                chromosome = self.population[i].copy()
                loc1 = np.random.randint(0, self.n)
                loc2 = np.random.randint(loc1, self.n)
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

        new_population = list()
        fitness_value_after_selection = list()
        sort_idx = np.argsort(fitness_value)
        for i in range(0, self.population_size):
            new_population.append(self.population[sort_idx[i]])
            fitness_value_after_selection.append(fitness_value[sort_idx[i]])

        self.population = new_population.copy()
        print("Best chromosome: %.4f" % np.min(fitness_value_after_selection))
        # print("Group average  : %.4f" % np.average(fitness_value_after_selection))

        best_sizes = self.compute_size_per_dcu(self.population[0])
        print(self.iter_cnt, fitness_value_after_selection[0], np.max(best_sizes) / np.average(best_sizes))

        if self.iter_cnt % 20 == 0:
            np.save('best_sizes_' + str(self.iter_cnt) + '.npy', best_sizes)
            plt.figure(figsize=(9, 5), dpi=200)
            plt.title('iter = %d, max / average = %.4f' % (self.iter_cnt, np.max(best_sizes) / np.average(best_sizes)))
            plt.plot(self.origin_size_per_dcu, color='blue', linestyle='--', alpha=0.3, label='before')
            plt.plot(best_sizes, color='blue', label='now')
            plt.legend(fontsize=15)
            plt.show()

    def evolve(self):
        self.crossover()
        self.mutate()
        self.select()
        self.iter_cnt += 1

    def show(self):
        pass

    def begin_evolution(self):
        self.form_initial_population()

        while self.iter_cnt < self.max_iter_times:
            # time1 = time.time()
            self.evolve()
            self.show()
            # time2 = time.time()
            # print('%.4f consumed.' %(time2 - time1))


if __name__ == "__main__":
    g = GeneticAlgorithm()
    g.begin_evolution()
    pass
