import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


def show_mat(mat, fig_name):
    plt.close()
    fig = plt.figure(figsize=(10, 10), dpi=400)
    ax = fig.add_subplot(111)
    ax.matshow(mat, cmap='binary')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(fig_name)
    plt.show()
    return


def group_the_conn_table(mat):
    binary_traffic_table_base_dcu = np.array(mat, dtype=bool)
    number_of_groups = 500
    n_dcu_per_group = 40
    binary_conn_table_after_grouping = np.zeros((number_of_groups, number_of_groups))

    print('Begin Calculating...')
    for row in range(number_of_groups):
        for col in range(number_of_groups):
            for i in range(n_dcu_per_group):
                temp = False
                row_idx = row * n_dcu_per_group + i
                for j in range(n_dcu_per_group):
                    col_idx = col * n_dcu_per_group + j
                    if binary_traffic_table_base_dcu[row_idx][col_idx] == 1:
                        binary_conn_table_after_grouping[row][col] = True
                        temp = True
                        break
                if temp:
                    break
        print(row)

    print(np.count_nonzero(binary_conn_table_after_grouping) / (number_of_groups**2))


def reverse_cuthill_mckee_and_show(mat):
    sort_array = ss.csgraph.reverse_cuthill_mckee(ss.coo_matrix(mat).tocsr(), symmetric_mode=True)
    for i in range(mat.shape[0]):
        mat[:, i] = mat[sort_array, i]
    for i in range(mat.shape[0]):
        mat[i, :] = mat[i, sort_array]

    show_mat(mat, 'reverse_cuthill_mckee_and_show.png')
    group_the_conn_table(mat[603:22603, 603:22603])
    return


def nested_dissection_ordering_and_show():
    pass


# N = 20000
# density = 0.02

# X = ss.random(N, N, density=density)
# X = (X + X.transpose()) / 2
# matrix = np.array(X.todense(), dtype=bool)

# matrix = np.load('../tables/traffic_table/traffic_table_base_dcu.npy')
path = '../tables/conn_table/conn_prob_v2.npz'

matrix = np.load(path)['conn_prob']

# matrix = ss.random(10, 10, density=0.1)
# matrix = coo_matrix(matrix).todense()
# matrix = (matrix + matrix.transpose()) / 2

matrix = np.array(matrix, dtype=bool)

# print(np.allclose(matrix, matrix.transpose()))
print(np.count_nonzero(matrix) / (matrix.shape[0]**2))

# show_mat(matrix, 'origin.png')
reverse_cuthill_mckee_and_show(matrix)
