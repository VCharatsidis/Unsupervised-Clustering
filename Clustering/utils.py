import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.datasets import fetch_openml




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

    print("string_to_numpy just finished "+str(filepath)+" !")
    return np.array(data)


def most_frequent(List):
    return max(set(List), key=List.count)


def miss_classifications(cluster):
    mfe = most_frequent(cluster)
    missclassifications = 0
    for j in cluster:
        if j != mfe:
            missclassifications += 1

    return missclassifications


def print_cluster_labels(cluster, members, targets):

    labels = [targets[int(members[m][0])+60000] for m in cluster]
    print("length: "+str(len(labels)))
    print(labels)
    missclassifications = miss_classifications(labels)
    print("miss classifications: "+str(missclassifications))
    print("")

    return missclassifications



