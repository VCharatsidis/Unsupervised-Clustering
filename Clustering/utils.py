import numpy as np
from matplotlib import pyplot as plt
import os

def show_mnist(first_image):
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def string_to_numpy(filepath):
    script_directory = os.path.split(os.path.abspath(__file__))[0]
    grubb = os.path.join(script_directory, filepath)

    f = open(grubb, "r")

    contents = f.readlines()
    data = []

    for line in contents:
        X = line.split(' ')
        input_x = []

        Z = map(float, X[:-1])

        for i in Z:
            input_x.append(i)

        data.append(np.array(input_x))

    print("string_to_numpy just finished!")
    return np.array(data)