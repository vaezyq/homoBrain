import gurobipy


def assignment(cost_matrix):
    # 保存行列标签
    index = cost_matrix.index
    columns = cost_matrix.columns

    # 创建模型
    model = gurobipy.Model('Assignment')
    x = model.addVars(index, columns, vtype=gurobipy.GRB.BINARY)
    t = model.addVar(vtype=gurobipy.GRB.INTEGER, name='t')
    model.update()

    # 设置目标函数
    model.setObjective(t)

    # 添加约束条件
    model.addConstr(gurobipy.quicksum(x[i, j] for i in index for j in columns) == min([len(index), len(columns)]))
    model.addConstrs(gurobipy.quicksum(x[i, j] for j in columns) <= 1 for i in index)
    model.addConstrs(gurobipy.quicksum(x[i, j] for i in index) <= 1 for j in columns)

    # 执行最优化
    model.optimize()

    # 输出信息
    result = cost_matrix * 0
    if model.status == gurobipy.GRB.Status.OPTIMAL:
        solution = [k for k, v in model.getAttr('x', x).items() if v == 1]
        for i, j in solution:
            print(f"{i} -> {j}：{cost_matrix.at[i,j]}")
            result.at[i, j] = 1
    return result


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    N = 2
    initial_solution = np.zeros((N, N), dtype=int)
    initial_solution[0][0] = 1
    initial_solution[0][1] = 2
    initial_solution[1][0] = 5
    initial_solution[1][1] = 10

    indexes = list([i for i in range(N)])

    cost_matrix = pd.DataFrame(initial_solution, index=indexes, columns=indexes)

    assignment(cost_matrix)
