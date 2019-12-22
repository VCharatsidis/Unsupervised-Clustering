from utils import string_to_numpy
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.datasets import fetch_openml
from utils import string_to_numpy, most_frequent, flatten
import os
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from Mutual_Information.train_MIM import to_Tensor
from ModelUnpredictability.train import calc_distance
import copy
from sklearn.cluster import KMeans
from ModelUnpredictability.one_thousand_brains import forward_block

filepath = '..\\ModelUnpredictability\\'
script_directory = os.path.split(os.path.abspath(__file__))[0]
detached_net_model = os.path.join(script_directory, filepath + 'detached_net.model')
conv = torch.load(detached_net_model)

colons = []
for i in range(676):
    path = 'one_thousand_brains\\predictor_' + str(i) + '.model'
    colon = os.path.join(script_directory, filepath + path)
    colons.append(torch.load(colon))

mnist = fetch_openml('mnist_784', version=1, cache=True)
targets = mnist.target[60000:]

X_train = mnist.data[:60000]
X_test = mnist.data[60000:]

fake_optimizers = []


def loss_representations(member_ids):
    with torch.no_grad():
        res = forward_block(X_test, member_ids, conv, colons, fake_optimizers, False)
        return res


def get_centroids(member_numbers):
    member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
    X = []
    for i in member_ids:
        X.append(loss_representations(i))

    # X1 = copy.deepcopy(X)
    # X1 = np.array(X1)
    # sort2 = X1.argsort(1)
    #
    # for c, i in enumerate(X):
    #     s = sort2[c].tolist()
    #     s = s[-200:]
    #
    #     s.sort()
    #     X[c] = [i[z] for z in s]
    #     print(X[c])
    #
    # X = np.array(X)
    # print(X.shape)

    predict = KMeans(n_clusters=10).fit_predict(X)

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


def log_distance_clustering(member_numbers):
    member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
    X = []
    for i in member_ids:
        X.append(loss_representations(i))

    X = np.array(X)
    miss = 0
    used = []

    for i in range(member_numbers):
        min = 100000
        min_idx = -1
        if i in used:
            continue
        for j in range(member_numbers):

            if i == j:
                continue

            log_dist = np.log(1-np.abs(X[i]-X[j]))

            sum = np.abs(log_dist.sum())

            if sum < min:
                min = sum
                min_idx = j

        used.append(min_idx)
        if targets[member_ids[i]] != targets[member_ids[min_idx]]:
            miss += 1

    print("missclassifications log distance best friend clustering: "+str(miss) + " percentage miss: "+str(miss/member_numbers))


log_distance_clustering(2000)
#get_centroids(1000)