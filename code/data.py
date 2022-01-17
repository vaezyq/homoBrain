import numpy as np
from base import Base


# 定义处理输入数据的类
class Data(Base):
    def __init__(self):
        super().__init__()

        # Biological data
        # 定义后续读取输入文件得到的结果，conn为连接矩阵
        self.conn = None
        # size为体素大小，origin_size为原式数据的size
        self.size, self.origin_size = None, None
        # 定义每个体素的度（与其它体素连接的个数）,origin_degree?
        self.degree, self.origin_degree = None, None
        # 定义连接矩阵的宽度，有两种，origin是原式的，n是去掉了一些的
        self.n, self.n_origin = None, None
        self.load_data()
        # 每个体素的大小乘以其相应的度，用作生成map?
        self.size_multi_degree = np.multiply(self.size, self.degree)

        # self.show_basic_information()

    def load_conn(self):
        if self.conn_version[0:5] == 'voxel':
            self.conn = np.load(self.conn_table_path)
        else:
            import pickle
            import sparse
            f = open(self.conn_root + 'conn.pickle', 'rb')
            self.conn = pickle.load(f)

    def load_size(self):
        self.size = np.load(self.size_path)
        if self.conn_version[0:5] == 'voxel':
            self.origin_size = self.size
        else:
            self.origin_size = np.load(self.origin_size_path)

    def load_degree(self):
        if self.conn_version[0:5] == 'voxel':
            self.degree = np.array([100] * self.n)
            # self.origin_degree = self.degree
        else:
            self.degree = np.load(self.degree_path)
            self.origin_degree = np.load(self.origin_degree_path)

    def load_data(self):
        self.load_conn()
        self.load_size()
        assert self.conn.shape[0] == self.size.shape[0]
        self.n = self.conn.shape[0]
        self.n_origin = self.origin_size.shape[0]
        self.load_degree()

    def show_basic_information(self):
        msg = ' Base Info '
        k_pounds = 19
        print('#' * k_pounds + msg + '#' * k_pounds)
        print("Number of GPUs used:", self.N)
        print("Number of voxels:", self.n)
        print("Number of neuron: %.2e" % self.neuron_number)
        print("Number of groups:", self.number_of_groups)
        print("Number of GPUs per group:", self.n_gpu_per_group)
        print("Connection table version:", self.conn_version)
        print("Map version:", self.map_version)
        print("Route version:", self.route_version)
        print("Sum size:", np.sum(self.size))
        print("Sum degree:", np.sum(self.degree))
        print('#' * (k_pounds * 2 + len(msg)))
        print()


if __name__ == '__main__':
    D = Data()
    D.load_data()
    print(D.conn)
    pass
