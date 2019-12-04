from utils import string_to_numpy
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.datasets import fetch_openml
import os


filepath = '..\\'
script_directory = os.path.split(os.path.abspath(__file__))[0]
encoder1_model = os.path.join(script_directory, filepath + 'encoder_1.model')
encoder_1 = torch.load(encoder1_model)

script_directory = os.path.split(os.path.abspath(__file__))[0]
encoder2_model = os.path.join(script_directory, filepath + 'encoder_2.model')
encoder_2 = torch.load(encoder1_model)

discriminator_model = os.path.join(script_directory, filepath + 'discriminator.model')
discriminator = torch.load(discriminator_model)
mnist = fetch_openml('mnist_784', version=1, cache=True)
targets = mnist.target

mnist = fetch_openml('mnist_784', version=1, cache=True)
targets = mnist.target

X_train = mnist.data[:60000]
X_test = mnist.data[60000:]


def write_mean_differences():
    read_file = "differences.txt"
    size = len(X_train)
    X = string_to_numpy(read_file)
    print(X.shape)
    means = np.mean(X, axis=1)
    print(means.shape)

    print("mean of means: "+ str(np.mean(means)))
    # file = open("mean_differences.txt", "a")
    # for i in means:
    #     file.writelines(str(i) + "\n")
    #
    # file.close()


#write_mean_differences()

def write_subgraph_differences(member_numbers):
    member_idx_to_id = {}
    with torch.no_grad():
        member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
        X_proto = X_test[member_ids, :]

        X_proto = np.reshape(X_proto, (member_numbers, 1, 28, 28))
        X_proto = Variable(torch.FloatTensor(X_proto))

        encoded_proto = encoder_2.forward(X_proto)

        for idx, id in enumerate(member_ids):
            member_idx_to_id[idx] = id
            ids = [id]
            X_image = X_test[ids, :]
            X_image = np.reshape(X_image, (1, 1, 28, 28))
            X_image = Variable(torch.FloatTensor(X_image))

            encoded_image = encoder_1.forward(X_image)
            encoded_image = encoded_image.repeat(member_numbers, 1)

            discriminator_input = torch.cat([encoded_image, encoded_proto], 1)
            discriminator_output = discriminator.forward(discriminator_input)

            disc = discriminator_output.detach().numpy()
            disc[idx] = 0
            file = open("subgraph_member_differences.txt", "a")
            for i in disc:
                file.writelines(str(i[0]) + ' ')

            file.writelines("\n")
            file.close()

        members_file = open("members.txt", "a")
        for m in member_ids:
            members_file.writelines(str(m)+" \n")
        members_file.close()

        return member_idx_to_id


def best_buddies_clustering():
    differences = string_to_numpy("subgraph_member_differences.txt")
    member_number = differences.shape[0]
    all_members = [x for x in range(member_number)]
    all_clusters = []
    used = []
    idx_to_cluster = {}

    cluster_counter = 0

    while len(all_members) > 0:
        cluster = []
        current_member = all_members[0]
        append_cluster = True

        while current_member not in cluster:

            if current_member in idx_to_cluster.keys():
                cluster_number = idx_to_cluster[current_member]
                all_clusters[cluster_number] += cluster
                append_cluster = False

                for k in idx_to_cluster.keys():
                    if idx_to_cluster[k] == cluster_counter:
                        idx_to_cluster[k] = cluster_number

                break;
            else:
                idx_to_cluster[current_member] = cluster_counter
                cluster.append(current_member)

            best_buddy_idx = np.argmax(differences[current_member])
            current_member = best_buddy_idx

        # print(cluster)
        # print("cluster made")

        if append_cluster:
            all_clusters.append(cluster)
            cluster_counter += 1

        used += cluster
        all_members = [x for x in range(member_number) if x not in used]

    print(len(all_clusters))

    for i in all_clusters:
        print(len(i))
        print(i)

    return all_clusters

def most_frequent(List):
    return max(set(List), key=List.count)


def display_clusters():
    clusters = best_buddies_clustering()
    X = string_to_numpy("members.txt")

    avg = 0
    total_misses = 0
    for c in clusters:
        print(" ")
        cluster_labels = [targets[int(X[m][0])+60000] for m in c]

        mfe = most_frequent(cluster_labels)
        counter = 0
        for i in cluster_labels:
            if i != mfe:
                counter += 1

        total_misses += counter
        cluster_size = len(cluster_labels)
        print("cluster size " + str(cluster_size))
        percentage_miss = (counter * 100) / cluster_size
        avg += percentage_miss
        print("clsuter: " + str(cluster_labels) + " mfe " + str(mfe) + " miss percentage " + str(percentage_miss))

    print("avg  miss: " + str(avg / len(clusters)))
    print("total miss: "+str(total_misses))
    print("total miss clusterings: "+str(total_misses / 3500))


member_idx_to_id = write_subgraph_differences(3500)
best_buddies_clustering()
display_clusters()