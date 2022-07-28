import numpy as np
import matplotlib.pyplot as plt
import os

path = "export_x/"
dir_list = os.listdir(path)
dir_list = [file[:-4] if file.endswith('.txt') else file for file in dir_list]
dir_list = [int(file) for file in dir_list]

dir_list.sort()
print(dir_list)

x = np.loadtxt("export/"+ str(dir_list[1]) + ".txt")

plt.ion()
figure = plt.imshow(x, cmap='RdBu',interpolation="none")  # cmap='hot', interpolation="nearest", cmap='hsv', interpolation="lanczos"
plt.axis('off')
plt.clim(0.2, 3.5)
plt.tight_layout()

def draw_figure(x):
    figure.set_data(x)
    #plt.draw()
    #plt.savefig("render/image" + str(h) + ".png", dpi=350)
    plt.pause(0.0001)
    return

for file in dir_list:
    draw_figure(np.loadtxt("export/" + str(file) + ".txt"))

    if ((file % 10) == 0):
        print(str(int(file/dir_list[-1]*100))+"%")
