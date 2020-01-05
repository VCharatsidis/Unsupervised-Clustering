import torch
import numpy as np
from torch.autograd import Variable
from sklearn.datasets import fetch_openml
from train_pixel import forward_block
import os
from sigmoid_layer import SigmoidLayer
import copy
from sklearn.cluster import KMeans
from train_pixel import to_Tensor
from base_conv import BaseConv
from train_pixel import show_mnist
from copy import deepcopy
from sklearn.metrics import mutual_info_score
from Pixel_prediction.train_pixel import neighbours

import torch


script_directory = os.path.split(os.path.abspath(__file__))[0]

encoder_model = os.path.join(script_directory, 'encoder.model')
encoder = torch.load(encoder_model)

decoder_model = os.path.join(script_directory, 'decoder.model')
decoder = torch.load(decoder_model)

mnist = fetch_openml('mnist_784', version=1, cache=True)
targets = mnist.target[60000:]

X_train = mnist.data[:60000]
X_test = mnist.data[60000:]

fake_optimizers = []



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
        list = []

        i = i / 255

        mean, std = encoder(i)

        #e = torch.zeros(mean.shape).normal_()
        z = std + mean

        output = decoder(z)
        show_mnist(i)
        show_mnist(output.detach().numpy())

        abs_difference = torch.abs(output - i)

        eps = 1e-8
        information_loss = torch.log(1 - abs_difference + eps)
        information_loss = torch.abs(information_loss)
        print(information_loss)
        information_loss = information_loss.detach().numpy()
        show_mnist(information_loss)

        indexes_of_biggest_elements = information_loss.argsort()[-5:][::-1]

        for i in range(784):
            if i not in indexes_of_biggest_elements:
                information_loss[i] = 0

        show_mnist(information_loss)

        X_kmeans.append(output.detach().numpy())

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


def log_distance_pairing(member_numbers):
    conv = SigmoidLayer()
    base_conv = BaseConv(1)

    member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
    X = X_test[member_ids]

    sigmoided = []
    for i in X:
        z = to_Tensor(i, 1)
        sigmoided.append(conv.forward(z).detach().numpy())

    print(sigmoided[0])
    X = np.array(sigmoided)

    # X = []
    # for i in member_ids:
    #     X. (loss_representations(i))
    #
    # X = np.array(X)

    miss = 0
    used = []

    print("done with representations")
    clusters = []
    for i in range(member_numbers):
        if i % 500 == 0:
            print("member number: "+str(i))

        min = 1000000
        min_idx = -1
        if i in used:
            continue

        for j in range(member_numbers):

            if i == j:
                continue

            log_dist = np.log(1 - np.abs(X[i] - X[j]))

            sum = np.abs(log_dist.sum())

            if sum < min:
                min = sum
                min_idx = j

        used.append(min_idx)
        if targets[member_ids[i]] != targets[member_ids[min_idx]]:
            miss += 1

    print("missclassifications log distance best friend clustering: " + str(miss) + " percentage miss: " + str(
        miss / member_numbers))


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


def log_distance_clustering(clusters, distances):

    new_clusters = []
    merged = []

    cluster_distances = calc_cluster_distances(clusters, distances)
    closest_clusters = closest_cluster_dictionary(clusters, cluster_distances)

    for idx, c in enumerate(clusters):
        if idx in merged:
            continue

        for idx2, c2 in enumerate(clusters):
            if idx >= idx2 or (idx2 in merged):
                continue

            if idx not in closest_clusters or idx2 not in closest_clusters:
                print(idx)
                print(idx2)
                input()

            if closest_clusters[idx] == idx2 and closest_clusters[idx2] == idx:
                for m2 in c2:
                    c.append(m2)

                merged.append(idx2)
                break

    for idx, c in enumerate(clusters):
        if idx not in merged:
            copied_cluster = copy.deepcopy(clusters[idx])
            new_clusters.append(copied_cluster)

    return new_clusters


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def calculate_distances_KL(data):
    distances = {}

    for idx, i in enumerate(data):
        for idx2, j in enumerate(data):
            if idx >= idx2:
                continue

            #sum = double_KL(i[1], j[1])
            sum = calc_MI(i[1], j[1], 150)
            print(sum)

            key1 = str(i[0]) + '_' + str(j[0])
            distances[key1] = sum

            key2 = str(j[0]) + '_' + str(i[0])
            distances[key2] = sum

    return distances


def double_KL(p, q):
    return KL_distance(p, q) + KL_distance(q, p)


def KL_distance(p, q):
    eps = 1e-8
    return np.sum(np.where(p != 0, p * np.log(p / q ), 0))


def calculate_distances(sigmoided_data):
    distances = {}
    eps = 1e-8
    for idx, i in enumerate(sigmoided_data):
        for idx2, j in enumerate(sigmoided_data):
            if idx >= idx2:
                continue

            distance = np.log(1 - np.abs(i[1] - j[1]) + eps)
            sum = np.abs(distance.sum())

            key1 = str(i[0])+'_'+str(j[0])
            distances[key1] = sum

            key2 = str(j[0])+'_'+str(i[0])
            distances[key2] = sum

    return distances


def calc_cluster_distances(clusters, distances):
    cluster_distances = {}
    for idx, c in enumerate(clusters):
        for idx2, c2 in enumerate(clusters):
            if idx >= idx2:
                continue

            distance = cluster_to_cluster_distance(c, c2, idx, idx2, distances)

            key1 = str(idx) + '_' + str(idx2)
            cluster_distances[key1] = distance

            key2 = str(idx2) + '_' + str(idx)
            cluster_distances[key2] = distance

    return cluster_distances


def cluster_to_cluster_distance(a, b, idx_a, idx_b, distances):
    cluster_sum = 0
    for m in a:
        member_sum = 0

        for m2 in b:
            if m[0] == m2[0]:
                print(idx_a)
                print(idx_b)
                print(len(a))
                print(len(b))
                input()
            key = str(m[0]) + '_' + str(m2[0])
            member_sum += distances[key]

        normalized = member_sum / len(b)
        cluster_sum += normalized

    return cluster_sum


def call_log_dist_clustering(member_numbers, cluster_number):
    member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
    X = X_test[member_ids]

    sigmoided = []
    idx_data = []

    # for counter, i in enumerate(X):
    #     list = []
    #     z = to_Tensor(i, 1)
    #     convolved = conv.forward(z).numpy()[0][0]
    #     image_and_index = (member_ids[counter], convolved)
    #
    #     idx_data.append(image_and_index)
    #     list.append(image_and_index)
    #     sigmoided.append(list)

    # for counter, i in enumerate(X):
    #     list = []
    #     ids = []
    #     ids.append(counter)
    #     ids = np.array(ids)
    #     res = np.array(loss_representations(ids))
    #     print(res.shape)
    #
    #     show_mnist(res)
    #
    #     image_and_index = (member_ids[counter], res)
    #
    #     idx_data.append(image_and_index)
    #     list.append(image_and_index)
    #     sigmoided.append(list)

    X = Variable(torch.FloatTensor(X))
    for counter, i in enumerate(X):
        list = []

        i = i/255

        mean, std = encoder(i)

        e = torch.zeros(mean.shape).normal_()
        z = std + mean

        generated_image = decoder(z)

        abs_difference = torch.abs(i - generated_image)
        eps = 1e-8
        L_reconstruction = torch.log(1 - abs_difference + eps)
        #L_reconstruction = torch.nn.functional.softmax(L_reconstruction)

        indexes_of_biggest_elements = L_reconstruction.detach().numpy().argsort()[:400][::-1]

        reshape = i.view(28, 28)

        inputs = neighbours(reshape)

        print(indexes_of_biggest_elements)
        indexes = np.array(indexes_of_biggest_elements)

        representation = [inputs[x].detach().numpy() for x in indexes]
        representation = np.array(representation)

        representation = representation.flatten()

        #print(representation)
        # show_mnist(i)
        # show_mnist(L_reconstruction.detach().numpy())

        image_and_index = (member_ids[counter], representation)
        idx_data.append(image_and_index)
        list.append(image_and_index)
        sigmoided.append(list)

        # output = decoder(z)
        #
        # image_and_index = (member_ids[counter], output.detach().numpy())
        # idx_data.append(image_and_index)
        # list.append(image_and_index)
        # sigmoided.append(list)

    print("done with sigmoid")
    #distances = calculate_distances(idx_data)
    distances = calculate_distances(idx_data)
    print("done with distances")

    counter = 0
    while len(sigmoided) > cluster_number:
        print("clustering iteration: "+str(counter) + " number clusters: "+str(len(sigmoided)))
        counter += 1
        sigmoided = log_distance_clustering(sigmoided, distances)

    total_miss = 0
    for c in sigmoided:
        labels = []
        for m in c:
            labels.append(targets[m[0]])

        miss = miss_classifications(labels)
        total_miss += miss
        print("most frequent: " + str(most_frequent(labels)))
        print("missclassifications: " + str(miss/len(labels)))

        print(labels)

    print("total missclassifications: " + str(total_miss/member_numbers))
    print("number classes: "+str(len(sigmoided)))




# call_log_dist_clustering(200, 15)
# log_distance_pairing(1000)
get_centroids(100)