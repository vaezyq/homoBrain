import imageio

path = "../../tables/traffic_table/0923_noon/"

images = list()
for i in range(10, 150, 10):
    images.append(imageio.imread(path + str(i) + ".png"))

imageio.mimsave(path + "hello.gif", images, fps=2)
