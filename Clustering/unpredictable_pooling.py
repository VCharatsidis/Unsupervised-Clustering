import torch
import numpy as np
from sklearn.datasets import fetch_openml
from utils import most_frequent, flatten
import os
from Mutual_Information.train_MIM import to_Tensor
from SimilarityMetric.train import calc_distance
from sklearn.cluster import KMeans

filepath = '..\\ModelUnpredictability\\'
script_directory = os.path.split(os.path.abspath(__file__))[0]
detached_net_model = os.path.join(script_directory, filepath + 'detached_net.model')
conv = torch.load(detached_net_model)

predictor_model = os.path.join(script_directory, filepath + 'predictor.model')
predictor = torch.load(predictor_model)

mnist = fetch_openml('mnist_784', version=1, cache=True)
targets = mnist.target

X_train = mnist.data[:60000]
X_test = mnist.data[60000:]


def unpredictable_representations(member_numbers, member_ids):
    with torch.no_grad():

        x_tensor = to_Tensor(X_test[member_ids, :], member_numbers)

        convolutions = conv.forward(x_tensor)

        x = flatten(convolutions)

        size = x.shape[1]

        loss = 0
        counter = 0

        # One hot encoding buffer that you create out of the loop and just keep reusing
        i_one_hot = torch.FloatTensor(member_numbers, size)

        total_loss = []

        for i in range(size):
            y = x[:, i]
            # In your for loop
            i_one_hot.zero_()

            i_tensor = torch.LongTensor(member_numbers, 1)
            i_tensor[:, 0] = i
            i_one_hot.scatter_(1, i_tensor, 1)

            x_reduced = torch.cat([x[:, 0:i], x[:, i + 1:]], 1)
            x_reduced = torch.cat([x_reduced, i_one_hot], 1)

            res = predictor.forward(x_reduced.detach())

            distances = calc_distance(res[0], y)
            loss += distances

            total_loss.append(distances.item())

            counter += 1

        return total_loss, loss


def Nmaxelements(list1, N):
    final_list = []

    for i in range(0, N):
        max1 = 0

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j];

        list1.remove(max1);
        final_list.append(max1)

    print(final_list)


def pooling(member_numbers):
    member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
    losses = []
    pixel_list = []

    for i in member_ids:

        loss, _ = unpredictable_representations(1, i)
        loss = np.array(loss)
        sort1 = loss.argsort()[-50:][::-1]
        sort1 = np.sort(sort1)
        #print(sort1)

        pixels = []
        for patch in sort1:
            row = patch % 26
            col = patch - row * 26

            for row_pos in range(3):
                for col_pos in range(3):
                    pixels.append(X_test[i, (row + row_pos) * 26 + col + col_pos])

        # print(pixels)
        # print(loss[sort1])
        # input()
        pixel_list.append(pixels)
        losses.append(loss[sort1])
        #losses.append(nlargest(15, loss))

    print(losses)

    return losses, member_ids, pixel_list


def get_centroids():
    targets = mnist.target[60000:]
    X, ids, pixels = pooling(200)
    print(pixels)

    #kmeans = KMeans(n_clusters=10).fit(X)
    predict = KMeans(n_clusters=10).fit_predict(pixels)

    # kmeans = KMeans(n_clusters=10).fit(X)
    # predict = KMeans(n_clusters=10).fit_predict(X)

    clusters_to_ids = {}
    true_labels_to_ids = {}
    for i, p in enumerate(predict):
        if p in clusters_to_ids.keys():
            clusters_to_ids[p].append(targets[ids[i]])
        else:
            idxs = []
            idxs.append(targets[ids[i]])
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
        percentage_miss = (counter*100) / cluster_size
        avg += percentage_miss
        print("clsuter: "+str(c)+" mfe " + str(mfe)+" miss percentage "+str(percentage_miss))

    print("avg  miss: "+str(avg/10))

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


get_centroids()