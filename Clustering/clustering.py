from sklearn.cluster import KMeans
from torchvision.utils import make_grid
from train import _read_raw_image_file
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import os
import matplotlib

import numpy as np

# X_train = _read_raw_image_file('..\\data\\raw\\binarized_mnist_train.amat')
# X_test = _read_raw_image_file('..\\data\\raw\\binarized_mnist_valid.amat')


def display_centroid(centroid, number):
    #sample = centroid.view(-1, 1, 28, 28)
    sample = centroid.reshape(28,28)
    sample = make_grid(sample, nrow=1).astype(np.float)
    matplotlib.image.imsave(f"centroid_{number}.png", sample)


mnist = fetch_openml('mnist_784', version=1, cache=True)
targets = mnist.target
print(targets[0])
print(mnist.data[0])
print(mnist.data[0].shape)
first_image = np.array(mnist.data[0], dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
display_centroid(mnist.data[0],11)
input()

def get_centroids(write_file, read_file):
    X = string_to_numpy(read_file)
    mnist = fetch_openml('mnist_784')
    labels = mnist.target
    print(len(labels))
    print(labels[0])
    print(labels[1])
    print(labels[2])
    print(labels[3])
    print(labels[4])
    print(labels[5])
    print(labels[6])
    print(labels[7])
    print(labels[8])
    print(labels[9])
    input()

    kmeans = KMeans(n_clusters=10).fit(X[:, :-1])
    predict = KMeans(n_clusters=10).fit_predict(X[:, :-1])
    print(predict[0])
    print(predict[1])
    print(predict[2])
    print(predict[3])
    print(predict[4])
    print(predict[5])
    print(predict[6])
    print(predict[7])
    print(predict[8])
    print(predict[9])
    print()

    input()

    print("centers gen: " + str(write_file))
    centroids = kmeans.cluster_centers_

    for centroid in range(10):
        file = open(str(write_file), "a")
        for i in centroids[centroid]:
            file.writelines(str(i) + ' ')

        file.writelines("\n")
        file.close()
        #display_centroid(centroids[centroid][0:2], centroid, write_file)





def string_to_numpy(filepath):
    script_directory = os.path.split(os.path.abspath(__file__))[0]
    grubb = os.path.join(script_directory, filepath)

    f = open(grubb, "r")

    contents = f.readlines()
    data = []

    for line in contents:
        X = line.split(' ')
        input_x = []

        if len(X) == 131:
            print(X)
            input()

        X[-3] = X[-2]

        Z = map(float, X[:-2])

        for i in Z:
            input_x.append(i)

        data.append(np.array(input_x))

    print("hi")
    return np.array(data)


get_centroids("centroids.txt", "encoded_reps.txt")
#
# def read_centroids():
#     list_centroids = []
#     centroids = string_to_numpy("centroids.txt")
#     for c in centroids:
#         list_centroids.append(centroids[-1])
#
#     return list_centroids
#

#
# print(X_train.shape)
# centroids = read_centroids()
# for c in centroids:
#     display_centroid(X_train[int(c)])



