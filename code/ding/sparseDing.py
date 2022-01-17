"""
用递推的方式，将带有二次约束的问题简化为MILP问题;
同时考虑N*N的以卡为单位的连接矩阵的稀疏性
详细的数学模型及推导过程见"20210618丁维洋老师关于组合优化问题的思考.pdf"
gurobi的使用方法参考
· https://zhuanlan.zhihu.com/p/52371462
· https://github.com/wurmen/Gurobi-Python

2021.6.27
"""

# # import gurobipy
# import time
# import numpy as np
#
#
# def optimize(A):
#     N = A.shape[0]
#
#     P = np.ones((N, N))
#
#     start_time = time.time()
#     # 计算alpha
#     alpha = np.zeros(N, dtype=int)
#     for i in range(N):
#         alpha[i] = np.sum(A[i, :])
#
#     # 创建模型
#     model = gurobipy.Model()
#     # model.setParam('OutputFlag', 0)
#     model.setParam("TimeLimit", 7200)
#
#     # 创建变量
#     t = model.addVar(vtype=gurobipy.GRB.INTEGER, name="t")
#     X = model.addVars(N, N, vtype=gurobipy.GRB.BINARY, name="X")
#
#     # 更新变量环境
#     model.update()
#
#     # 创建目标函数
#     model.setObjective(t, gurobipy.GRB.MINIMIZE)
#
#     # 创建约束条件
#     model.addConstrs(X.sum(i, "*") <= (t - alpha[i]) for i in range(N))
#     model.addConstrs(
#         sum(A[i][k] * X[k, j] + X[i, k] * A[k][j] for k in range(N)) >= 1 for i in range(N) for j in range(N))
#
#     # # 考虑连接稀疏性的情形
#     # model.addConstrs(
#     #     sum(A[i][k] * X[k, j] + X[i, k] * A[k][j] for k in range(N)) >= P[i][j] for i in range(N) for j in range(N))
#
#     # for i in range(N):
#     #     for j in range(N):
#     #         temp = 0
#     #         for k in range(N):
#     #             temp += A[i][k] * X[k, j] + X[i, k] * A[k][j]
#     #         model.addConstr(temp >= 1)
#
#     # 执行线性规划模型
#     model.optimize()
#
#     # 输出模型结果
#     result = int(model.objVal)
#     # print("Min link number: {:.0f}".format(model.objVal))
#
#     end_time = time.time()
#     print("########################")
#     print("N = %d, %.2fs consumed" % (2 * N, end_time - start_time))
#     print("########################")
#
#     # 把输出转化为np.array()
#     solution = np.zeros((2 * N, 2 * N), dtype=int)
#     for i in range(N):
#         for j in range(N):
#             solution[i][j], solution[i + N][j + N] = A[i][j], A[i][j]
#
#     for i in range(N):
#         for j in range(N, 2 * N):
#             solution[i][j], solution[j][i] = X[i, j - N].x, X[i, j - N].x
#
#     # print(solution)
#
#     draw_and_save(solution, result, show_graph=True)
#
#     assert check(solution)
#
#
# def check(matrix):
#     return (np.matmul(matrix, matrix) >= 1).all()
#
#
# def draw_and_save(matrix, t, show_graph=False):
#     import networkx as nx
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#
#     N = matrix.shape[0]
#     graph = nx.from_numpy_matrix(matrix)
#     node_location = nx.circular_layout(graph)
#     plt.figure(figsize=(18, 8))
#
#     plt.subplot(121)
#     ax = plt.gca()
#     ax.set_title('N = %d, link num = %d' % (N, t))
#     node_size = 330 if N < 40 else (10000 / N)
#     nx.draw(graph, pos=node_location, with_labels=False, node_size=node_size, node_color='green', ax=ax)
#
#     plt.subplot(122)
#     plt.title('N = %d, link num = %d' % (N, t))
#     sns.heatmap(matrix, cbar=False, cmap='binary')
#
#     fig_save_path = '../graph/' + str(N) + '.png'
#     plt.savefig(fig_save_path)
#     print("%d.png saved." % N)
#
#     txt_save_path = '../graph/' + str(N) + '.txt'
#     np.savetxt(txt_save_path, matrix, fmt='%d')
#     print("%d.txt saved." % N)
#
#     if show_graph:
#         plt.show()
#
#
# number_of_dcu = 5
# path = "../graph/"
#
# while number_of_dcu < 2000:
#     print("Begin optimizing N = %d..." % number_of_dcu)
#     initial_solution = np.loadtxt(path + str(number_of_dcu) + '.txt', dtype=int)
#     number_of_dcu *= 2
#     optimize(initial_solution)
