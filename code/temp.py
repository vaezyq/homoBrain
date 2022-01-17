import numpy as np
import matplotlib.pyplot as plt

desktop_path = "C:/Users/liuyuhao/Desktop/"
folder = "1207_noon2/"

for i in range(10, 30, 10):
    data = np.load(desktop_path + folder + "traffic_out" + str(i) + ".npy")
    plt.figure(figsize=(10, 6), dpi=200)
    plt.ylim(-0.1e9, 1.6e9)
    plt.title("%d: max/average =%.4f" % (i, np.max(data) / np.average(data)))
    X = np.arange(0, 10000)
    y_average = np.full(10000, np.average(data))
    plt.plot(X, data, color='blue')
    plt.plot(X, y_average, color='red')
    plt.savefig(desktop_path + folder + "figures/out" + str(i) + ".png")
    plt.clf()
