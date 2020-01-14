import torch
import numpy as np
from torch.autograd import Variable
from sklearn.datasets import fetch_openml
from train_MIM import get_data
import os
import copy
from sklearn.cluster import KMeans
from train_pixel import to_Tensor
from train_pixel import show_mnist
import statistics


script_directory = os.path.split(os.path.abspath(__file__))[0]

base_conv_model = os.path.join(script_directory, 'mi_encoder_1.model')
encoder = torch.load(base_conv_model)

decoder_model = os.path.join(script_directory, 'mi_discriminator.model')
discriminator = torch.load(decoder_model)

mnist = fetch_openml('mnist_784', version=1, cache=True)
targets = mnist.target[60000:]

X_train = mnist.data[:60000]
X_test = mnist.data[60000:]

fake_optimizers = []


def flatten(out):
    return out.view(out.shape[0], -1)


def pair_forward(encoded_image, encoded_image_2):
    mlp_in = torch.cat([encoded_image, encoded_image_2], 1)
    mlp_out = discriminator(mlp_in)

    return mlp_out


def get_results(counter, results, encoded_1, encoded_2):
    for i in encoded_1:
        for j in encoded_2:
            results[counter] = pair_forward(i, j)
            counter += 1

    return counter, results


def similarity_rep(encodings, id_2):
    with torch.no_grad():
        sample_2 = X_test[id_2]
        x = np.array([sample_2])
        x2 = get_data(x)

        encodings_2 = []

        for image in x2:
            out = encoder(image)
            out_flattened = flatten(out)
            encodings_2.append(out_flattened)

        counter = 0
        combinations = 49
        results = torch.zeros(combinations)

        counter, results = get_results(counter, results, encodings, encodings_2)

        return results


def miss_classifications(cluster):
    mfe = most_frequent(cluster)
    missclassifications = 0
    for j in cluster:
        if j != mfe:
            missclassifications += 1

    return missclassifications


def get_centroids(member_numbers):
    member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
    X = []
    for i in member_ids:
        res = loss_representations(i, member_ids)
        print(res)
        print(len(res))
        X.append(res)

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
        avg += counter
        print("clsuter: " + str(c) + " mfe " + str(mfe) + " miss percentage " + str(percentage_miss))

    print("avg  miss: " + str(avg / member_numbers))

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
    #     X.append(loss_representations(i))
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
        min_distance = 100000000
        for idx2, c2 in enumerate(clusters):
            if idx == idx2:
                continue

            key = str(idx) + '_' + str(idx2)
            distance = cluster_distances[key]

            if distance < min_distance:
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


def calculate_distances(sigmoided_data):
    distances = {}
    for idx, i in enumerate(sigmoided_data):
        for idx2, j in enumerate(sigmoided_data):
            if idx >= idx2:
                continue

            absolute_differece = np.abs(i[1] - j[1])

            zeros = np.zeros(absolute_differece.shape[0])
            ones = np.ones(absolute_differece.shape[0])

            absolute_differece = np.maximum(zeros, absolute_differece)
            absolute_differece = np.minimum(ones, absolute_differece)

            eps = 1e-8
            distance = np.log(1 - absolute_differece+eps)
            sum = np.abs(distance.sum())

            key1 = str(i[0])+'_'+str(j[0])
            distances[key1] = sum

            key2 = str(j[0])+'_'+str(i[0])
            distances[key2] = sum

    return distances


def loss_representations(id, member_ids):
    with torch.no_grad():
        x_train = X_test[id]

        x = np.array([x_train])
        x1 = get_data(x)

        encodings = []

        for image in x1:
            out = encoder(image)
            out_flattened = flatten(out)
            encodings.append(out_flattened)

        similarities = []

        for i in member_ids:
            rep = similarity_rep(encodings, i)
            # print(rep)
            mean_rep = torch.mean(rep)

            mean_rep = mean_rep.detach().numpy()

            similarities.append(mean_rep)

        return similarities



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

    for counter, i in enumerate(X):
        list = []
        ids = []
        ids.append(member_ids[counter])
        ids = np.array(ids)


        res = loss_representations(member_ids[counter], member_ids)
        res = np.array(res)

        # loss, _, res, _ = loss_representations(ids)
        # print("res shape", res.shape)
        #
        # res = np.array(res)
        # res = np.reshape(res, (1, 1, 28, 28))
        # res = Variable(torch.FloatTensor(res))
        #
        # i = np.array(i)
        # i = np.reshape(i, (1, 1, 28, 28))
        # i = Variable(torch.FloatTensor(i))
        #
        # loss = np.array(loss)
        #
        # show_mnist(i)
        # show_mnist(res)

        # indexes_of_biggest_elements = loss.argsort()[-100:][::-1]
        # print(indexes_of_biggest_elements)
        #
        # for c, image in enumerate(loss):
        #     if c not in indexes_of_biggest_elements:
        #         loss[c] = 0



        # abs_difference = torch.abs(i - res)
        # eps = 1e-8
        # L_reconstruction = torch.log(1 - abs_difference + eps)
        # show_mnist(L_reconstruction)


        # loss = np.reshape(loss, (1, 1, 28, 28))
        # loss = Variable(torch.FloatTensor(loss))

        image_and_index = (member_ids[counter], res)

        idx_data.append(image_and_index)
        list.append(image_and_index)
        sigmoided.append(list)

    print("done with sigmoid")
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


def visualise(label):
    same_labels = []
    for c, i in enumerate(targets):
        print(i)
        #if i == label:
        same_labels.append(c)
        if len(same_labels) > 10:
            break


    for i in same_labels:
        res = loss_representations(i, same_labels)

        print(res)
        show_mnist(X_test[i])


#visualise('5')
call_log_dist_clustering(50, 12)
# log_distance_pairing(1000)
#get_centroids(100)