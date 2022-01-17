import gurobipy
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_and_save(number_of_dcu):
    # 创建模型
    model = gurobipy.Model()
    model.setParam('OutputFlag', 0)
    # model.setParam("TimeLimit", 5*60)
    
    # 创建变量
    t = model.addVar(vtype=gurobipy.GRB.INTEGER, name="t")
    A = model.addVars(number_of_dcu, number_of_dcu, vtype=gurobipy.GRB.BINARY, name="A")
    
    # 更新变量环境
    model.update()
    
    # 创建目标函数
    model.setObjective(t, gurobipy.GRB.MINIMIZE)
    
    # # 创建约束条件
    # 线性约束
    model.addConstrs(A[i, i] == 1 for i in range(number_of_dcu))
    model.addConstrs(A[i, j] == A[j, i] for i in range(number_of_dcu) for j in range(number_of_dcu))
    
    # 边界约束
    model.addConstrs(A.sum(i, "*") <= t for i in range(number_of_dcu))
    
    # 二次约束
    # model.addConstrs((sum(A[i, k] * A[k, j]) for k in range(number_of_dcu)) <= t for i in range(number_of_dcu) for j in range(number_of_dcu))
    for i in range(number_of_dcu):
        for j in range(number_of_dcu):
            temp = 0
            for k in range(number_of_dcu):
                temp += A[i, k] * A[k, j]
            model.addConstr(temp >= 1)
    
    # 执行线性规划模型
    model.optimize()
    
    # 输出模型结果
    result = int(model.objVal)
    print("Min link number: {:.0f}".format(model.objVal))
    
    # 把输出转化为np.array()
    matrix = np.zeros((number_of_dcu, number_of_dcu), dtype=int)
    for i in range(number_of_dcu):
        for j in range(number_of_dcu):
            matrix[i][j] = A[i, j].x
    
    # print("\nSolution matrix: ")
    # for i in range(number_of_dcu):
    #     for j in range(number_of_dcu):
    #         print("{:.0f}".format(0 if (A[i, j].x == -0.0) else matrix[i][j]), end="  ")
    #     print()

    # 将解保存为txt形式
    txt_save_path = 'C:/all/WOW/brain/partition_and_route/graph/' + str(number_of_dcu) + '.txt'
    np.savetxt(txt_save_path, matrix, fmt='%d')

    # 画图并保存
    graph = nx.from_numpy_matrix(matrix)
    node_location = nx.circular_layout(graph)
    plt.figure(figsize=(18, 8))

    plt.subplot(121)
    ax = plt.gca()
    ax.set_title('N = %d, link num = %d' % (number_of_dcu, result))
    nx.draw(graph, pos=node_location, with_labels=False, node_color='green', ax=ax)

    plt.subplot(122)
    plt.title('N = %d, link num = %d' % (number_of_dcu, result))
    sns.heatmap(matrix, cbar=False, cmap='binary')

    fig_save_path = 'C:/all/WOW/brain/partition_and_route/graph/' + str(number_of_dcu) + '.png'
    plt.savefig(fig_save_path)


time_lst = list()
for i in range(4, 40):
    start_time = time.time()
    print('###########################')
    print('Begin N = %d' % i)
    draw_and_save(i)
    end_time = time.time()
    print('%.4f seconds consumed.' % (end_time-start_time))
    time_lst.append(end_time-start_time)


time_lst = np.array(time_lst)
np.savetxt('C:/all/WOW/brain/partition_and_route/time.txt', time_lst, fmt='%d')
print(time_lst)
