
from sklearn.datasets import fetch_openml
import os
from sklearn.cluster import KMeans

from stl10_input import read_all_images, read_labels
from stl_utils import *
from sklearn.metrics import mutual_info_score

import torch


script_directory = os.path.split(os.path.abspath(__file__))[0]

fileName = "..\\data\\stl10_binary\\train_X.bin"
X_train = read_all_images(fileName)

train_y_File = "..\\data\\stl10_binary\\train_y.bin"
y_train = read_labels(train_y_File)
y_train = np.array([x % 10 for x in y_train])

testFile = "..\\data\\stl10_binary\\test_X.bin"
X_test = read_all_images(testFile)

test_y_File = "..\\data\\stl10_binary\\test_y.bin"
targets = read_labels(test_y_File)
targets = np.array([x % 10 for x in targets])

encoder_name = "..\\binary_brain_stl_mean\\stl_binary"

encoder = torch.load(encoder_name+".model")
encoder.eval()



def miss_classifications(cluster):
    mfe = most_frequent(cluster)
    missclassifications = 0
    for j in cluster:
        if j != mfe:
            missclassifications += 1

    return missclassifications


def get_centroids(member_numbers):
    member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
    X = X_test[member_ids]
    X_kmeans = []
    # for i in member_ids:
    #     X.append(loss_representations([i]))

    X = Variable(torch.FloatTensor(X))
    for counter, i in enumerate(X):
        x_tensor = to_tensor(images)

        x_train = rgb2gray(x_tensor)
        x_tensor = to_tensor(x_train)
        x_tensor = x_tensor.unsqueeze(0)
        images = x_tensor.transpose(0, 1)
        images = images.transpose(2, 3)

        images = images / 255.0

        pad = (96 - SIZE) // 2
        images = scale(images, SIZE, pad, BATCH_SIZE_DEFAULT)
        images = images[:, :, pad:96 - pad, pad:96 - pad]

        X_kmeans.append(mean.detach().numpy())


    predict = KMeans(n_clusters=10).fit_predict(X_kmeans)

    clusters_to_ids = {}

    for i, p in enumerate(predict):
        if p in clusters_to_ids.keys():
            clusters_to_ids[p].append(targets[member_ids[i]])
        else:
            idxs = []
            idxs.append(targets[member_ids[i]])
            clusters_to_ids[p] = idxs

    avg = 0
    for c in clusters_to_ids.keys():
        mfe = most_frequent(clusters_to_ids[c])
        counter = 0
        for i in clusters_to_ids[c]:
            if i != mfe:
                counter += 1

        cluster_size = len(clusters_to_ids[c])
        print("cluster size " + str(cluster_size))
        percentage_miss = (counter * 100) / cluster_size
        avg += percentage_miss
        print("clsuter: " + str(c) + " mfe " + str(mfe) + " miss percentage " + str(percentage_miss))

    print("avg  miss: " + str(avg / 10))

    print(clusters_to_ids[0])
    print(clusters_to_ids[1])
    print(clusters_to_ids[2])
    print(clusters_to_ids[3])
    print(clusters_to_ids[4])
    print(clusters_to_ids[5])
    print(clusters_to_ids[6])
    print(clusters_to_ids[7])
    print(clusters_to_ids[8])
    print(clusters_to_ids[9])


def most_frequent(List):
    return max(set(List), key=List.count)


def closest_cluster_dictionary(clusters, cluster_distances):
    closest_cluster = {}
    for idx, c in enumerate(clusters):
        closest_cluster[idx] = -1
        min_distance = -100000000
        for idx2, c2 in enumerate(clusters):
            if idx == idx2:
                continue

            key = str(idx) + '_' + str(idx2)
            distance = cluster_distances[key]

            if distance > min_distance:
                min_distance = distance
                closest_cluster[idx] = idx2

    return closest_cluster


get_centroids(300)