from utils import string_to_numpy
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.datasets import fetch_openml
import utils
import os
from torchvision import transforms
import torchvision.transforms.functional as F
import random


filepath = '..\\'
script_directory = os.path.split(os.path.abspath(__file__))[0]
encoder1_model = os.path.join(script_directory, filepath + 'encoder_1.model')
encoder_1 = torch.load(encoder1_model)

encoder2_model = os.path.join(script_directory, filepath + 'encoder_2.model')
encoder_2 = torch.load(encoder2_model)

discriminator3_model = os.path.join(script_directory, filepath + 'discriminator3.model')
discriminator3 = torch.load(discriminator3_model)

discriminator4_model = os.path.join(script_directory, filepath + 'discriminator4.model')
discriminator4 = torch.load(discriminator4_model)

discriminator5_model = os.path.join(script_directory, filepath + 'discriminator5.model')
discriminator5 = torch.load(discriminator5_model)

discriminator6_model = os.path.join(script_directory, filepath + 'discriminator6.model')
discriminator6 = torch.load(discriminator6_model)

meta_discriminator_model = os.path.join(script_directory, filepath + 'meta_discriminator.model')
meta_discriminator = torch.load(meta_discriminator_model)

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
def flatten(out):
     return out.view(out.shape[0], -1)


def write_subgraph_differences(member_numbers):
    member_idx_to_id = {}
    with torch.no_grad():
        member_ids = np.random.choice(len(X_test), size=member_numbers, replace=False)
        X_proto_noise = X_test[member_ids, :]
        X_proto_rotate = X_test[member_ids, :]
        X_proto_scale = X_test[member_ids, :]

        threshold = random.uniform(0, 0.35)
        for i in range(X_proto_noise.shape[0]):
            nums = np.random.uniform(low=0, high=1, size=(X_proto_noise[i].shape[0],))
            X_proto_noise[i] = np.where(nums > threshold, X_proto_noise[i], 0)

        X_proto_noise = np.reshape(X_proto_noise, (member_numbers, 1, 28, 28))
        X_proto_noise = Variable(torch.FloatTensor(X_proto_noise))

        X_proto_rotate = np.reshape(X_proto_rotate, (member_numbers, 1, 28, 28))
        X_proto_rotate = Variable(torch.FloatTensor(X_proto_rotate))

        X_proto_scale = np.reshape(X_proto_scale, (member_numbers, 1, 28, 28))
        X_proto_scale = Variable(torch.FloatTensor(X_proto_scale))

        for i in range(X_proto_rotate.shape[0]):
            transformation = transforms.RandomRotation(10)
            trans = transforms.Compose(
                [transformation, transforms.ToTensor()])

            a = F.to_pil_image(X_proto_rotate[i])
            trans_image = trans(a)
            X_proto_rotate[i] = trans_image

        for i in range(X_proto_scale.shape[0]):
            transformation = transforms.Resize(20, interpolation=2)
            trans = transforms.Compose(
                [transformation, transforms.Pad(4), transforms.ToTensor()])

            a = F.to_pil_image(X_proto_scale[i])
            trans_image = trans(a)
            X_proto_scale[i] = trans_image

        out_noise_6 = encoder_2.forward(X_proto_noise)
        out_rotate_6 = encoder_2.forward(X_proto_rotate)
        out_scale_6 = encoder_2.forward(X_proto_scale)

        print(X_proto_scale[0])
        print(X_proto_noise[0])
        print(X_proto_rotate[0])
        input()

        out_noise_6 = flatten(out_noise_6)
        out_rotate_6 = flatten(out_rotate_6)
        out_scale_6 = flatten(out_scale_6)

        for idx, id in enumerate(member_ids):
            member_idx_to_id[idx] = id
            ids = [id]
            X_image = X_test[ids, :]
            X_image = np.reshape(X_image, (1, 1, 28, 28))
            X_image = Variable(torch.FloatTensor(X_image))

            out_6 = encoder_1.forward(X_image)
            out_6 = flatten(out_6)

            #out_3 = out_3.repeat(member_numbers, 1)

            out_6 = out_6.repeat(member_numbers, 1)

            #discriminator3_input = torch.cat([out_3, out_2_3], 1)

            discriminator6_noise = torch.cat([out_6, out_noise_6], 1)
            discriminator6_rotate = torch.cat([out_6, out_rotate_6], 1)
            discriminator6_scale = torch.cat([out_6, out_scale_6], 1)

            #discriminator3_output = discriminator3.forward(discriminator3_input)
            # discriminator4_output = discriminator4.forward(discriminator4_input)
            # discriminator5_output = discriminator5.forward(discriminator5_input)
            # discriminator6_output = discriminator6.forward(discriminator6_input)

            disc_noise = meta_discriminator.forward(discriminator6_noise)
            disc_rotate = meta_discriminator.forward(discriminator6_rotate)
            disc_scale = meta_discriminator.forward(discriminator6_scale)

            discriminator_output = (disc_noise + disc_rotate + disc_scale)/3

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
    X = string_to_numpy("members.txt")
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
                for k in cluster:
                    idx_to_cluster[k] = cluster_number
                break;
            else:
                cluster.append(current_member)

            best_buddy_idx = np.argmax(differences[current_member])
            current_member = best_buddy_idx

        # print(cluster)
        # print("cluster made")

        if append_cluster:
            for k in cluster:
                idx_to_cluster[k] = cluster_counter

            all_clusters.append(cluster)
            cluster_counter += 1

        used += cluster
        all_members = [x for x in range(member_number) if x not in used]

    print(len(all_clusters))

    for i in all_clusters:
        utils.print_cluster_labels(i, X)

    return all_clusters


def pair_clustering():
    differences = string_to_numpy("subgraph_member_differences.txt")
    X = string_to_numpy("members.txt")
    pair_clusters = []
    used = []
    member_number = differences.shape[0]

    avg = 0
    total_misses = 0

    for i in range(member_number):
        best_buddy_idx = np.argmax(differences[i])
        if best_buddy_idx in used:
            continue

        if np.argmax(differences[best_buddy_idx]) == i:
            cluster = []
            cluster.append(i)
            cluster.append(best_buddy_idx)
            used.append(i)
            used.append(best_buddy_idx)
            pair_clusters.append(cluster)

    for c in pair_clusters:
        cluster_labels = [targets[int(X[m][0])+60000] for m in c]
        mfe = utils.most_frequent(cluster_labels)
        counter = 0
        for i in cluster_labels:
            if i != mfe:
                counter += 1

        if counter > 0:
            utils.show_mnist(X_test[int(X[c[0]][0])])
            utils.show_mnist(X_test[int(X[c[1]][0])])

        total_misses += counter
        cluster_size = len(cluster_labels)
        print("cluster size " + str(cluster_size))
        percentage_miss = (counter * 100) / cluster_size
        avg += percentage_miss
        print("clsuter: " + str(cluster_labels) + " mfe " + str(mfe) + " miss percentage " + str(percentage_miss))


member_idx_to_id = write_subgraph_differences(1000)
#best_buddies_clustering()
pair_clustering()

def get_best_friends():
    differences = string_to_numpy("subgraph_member_differences.txt")
    members_number = differences.shape[0]
    members = [x for x in range(members_number)]
    best_friends = {}
    for m in members:
        best_buddy_idx = np.argmax(differences[m])
        best_friends[m] = best_buddy_idx

    return best_friends


#display_clusters()
