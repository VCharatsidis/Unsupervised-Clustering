from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from utils import string_to_numpy, most_frequent


# def display_centroid(centroid, number):
#     #sample = centroid.view(-1, 1, 28, 28)
#     sample = centroid.reshape(28, 28)
#     sample = make_grid(sample, nrow=1).astype(np.float)
#     matplotlib.image.imsave(f"centroid_{number}.png", sample)
#
#
# mnist = fetch_openml('mnist_784', version=1, cache=True)
# targets = mnist.target
# print(targets[0])
# print(mnist.data[0])
# print(mnist.data[0].shape)
# first_image = np.array(mnist.data[0], dtype='float')
#
# display_centroid(mnist.data[0],11)
# input()


# = mnist.data[:60000]
# = mnist.data[60000:]


def get_centroids(write_file, read_file):
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target
    X = string_to_numpy(read_file)
    mnist = fetch_openml('mnist_784')
    labels = mnist.target

    kmeans = KMeans(n_clusters=10).fit(X)
    predict = KMeans(n_clusters=10).fit_predict(X)

    # kmeans = KMeans(n_clusters=10).fit(X)
    # predict = KMeans(n_clusters=10).fit_predict(X)

    clusters_to_ids = {}
    true_labels_to_ids = {}
    for i, p in enumerate(predict):
        if p in clusters_to_ids.keys():
            clusters_to_ids[p].append(targets[i])
        else:
            idxs = []
            idxs.append(targets[i])
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
        print("clsuter: "+str(c)+" mfe " +str(mfe)+" miss percentage "+str(percentage_miss))

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

    print(" ")

    print(str(labels[1]) + " " + str(predict[1]))
    print(str(labels[2]) + " " + str(predict[2]))
    print(str(labels[3]) + " " + str(predict[3]))
    print(str(labels[4]) + " " + str(predict[4]))
    print(str(labels[5]) + " " + str(predict[5]))
    print(str(labels[6]) + " " + str(predict[6]))
    print(str(labels[7]) + " " + str(predict[7]))
    print(str(labels[8]) + " " + str(predict[8]))
    print(str(labels[9]) + " " + str(predict[9]))
    print(str(labels[10]) + " " + str(predict[10]))
    print(str(labels[11]) + " " + str(predict[11]))
    print(str(labels[12]) + " " + str(predict[12]))
    print(str(labels[13]) + " " + str(predict[13]))
    print(str(labels[14]) + " " + str(predict[14]))
    print(str(labels[15]) + " " + str(predict[15]))
    print(str(labels[16]) + " " + str(predict[16]))
    print(str(labels[17]) + " " + str(predict[17]))
    print(str(labels[18]) + " " + str(predict[18]))
    print(str(labels[19]) + " " + str(predict[19]))
    print(str(labels[20]) + " " + str(predict[20]))

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









#show_mnist()
#get_centroids("centroids.txt", "differences.txt")
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



