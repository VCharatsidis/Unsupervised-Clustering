from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import scipy.io as sio
import argparse
import os

from one_hot_net import OneHotNet

from stl_utils import *
import random

from torchvision.utils import make_grid
import torch.nn.functional as F
import matplotlib
import pickle

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(sci_mode=False)

from torchvision import models
EPS = sys.float_info.epsilon

#EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4

MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 512

#INPUT_NET = 3072
INPUT_NET = 5120
SIZE = 32
SIZE_Y = 32
NETS = 1

CLASSES = 100
DESCRIPTION = " Image size: " + str(SIZE) + " , Classes: " + str(CLASSES)

EVAL_FREQ_DEFAULT = 250
MIN_CLUSTERS_TO_SAVE = 100
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None


cluster_accuracies = {}
for i in range(CLASSES):
    cluster_accuracies[i] = 0


class_numbers = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 0: 0}


transformations_dict = {0: "original",
                       1: "scale",
                       2: "rotate",
                       3: "reverse pixel value",
                       4: "sobel total",
                       5: "sobel x",
                       6: "sobel y",
                       7: "gaussian blur",
                       8: "random crop",
                       9: "random crop",
                       10: "random crop",
                       11: "random crop",
                       12: "image 1",
                       13: "image 2"}


labels_to_imags = {}
for i in range(CLASSES):
    labels_to_imags[i] = i


def save_images(images, transformation):
    print(transformations_dict[transformation])
    numpy_cluster = images.cpu().detach()
    save_cluster(numpy_cluster, transformations_dict[transformation], 0)


def another(a, b, total_mean):
    batch_mean = (a.mean(dim=0) + b.mean(dim=1)) / 2
    total_mean = total_mean.cuda() * 0.99 + 0.01 * batch_mean

    batch_sum = (a.sum(dim=0) + b.sum(dim=0)) / 2

    product = (a * b) / batch_sum
    product_individual_mean = product.mean(dim=1)

    log = - torch.log(product_individual_mean)

    scalar = log.mean()

    return scalar, total_mean


def entropy_minmax_loss(a, b, total_mean):
    batch_mean = ((a.mean(dim=0) + b.mean(dim=0)) / 2)
    total_mean = total_mean.cuda() * 0.95 + 0.05 * batch_mean

    penalty = (a.sum(dim=0) + b.sum(dim=0)) / 2

    p1 = (a * b) / penalty
    p1 = p1.sum(dim=1)

    log = - torch.log(p1)

    scalar = log.mean()

    return scalar, total_mean


def forward_block(X, ids, encoder, optimizer, train, total_mean):
    number_transforms = 17
    aug_ids = np.random.choice(number_transforms, size=number_transforms, replace=False)

    image = X[ids, :]

    eight = image.shape[0] // 8

    image_1 = transformation(aug_ids[0], image[0:eight], SIZE, SIZE_Y)
    image_2 = transformation(aug_ids[1], image[0:eight], SIZE, SIZE_Y)

    image_3 = transformation(aug_ids[2], image[eight: 2 * eight], SIZE, SIZE_Y)
    image_4 = transformation(aug_ids[3], image[eight: 2 * eight], SIZE, SIZE_Y)

    image_5 = transformation(aug_ids[4], image[2 * eight: 3 * eight], SIZE, SIZE_Y)
    image_6 = transformation(aug_ids[5], image[2 * eight: 3 * eight], SIZE, SIZE_Y)

    image_7 = transformation(aug_ids[6], image[3 * eight: 4 * eight], SIZE, SIZE_Y)
    image_8 = transformation(aug_ids[7], image[3 * eight: 4 * eight], SIZE, SIZE_Y)

    image_9 = transformation(aug_ids[8], image[4 * eight: 5 * eight], SIZE, SIZE_Y)
    image_10 = transformation(aug_ids[9], image[4 * eight: 5 * eight], SIZE, SIZE_Y)

    image_11 = transformation(aug_ids[10], image[5 * eight: 6 * eight], SIZE, SIZE_Y)
    image_12 = transformation(aug_ids[11], image[5 * eight: 6 * eight], SIZE, SIZE_Y)

    image_13 = transformation(aug_ids[12], image[6 * eight: 7 * eight], SIZE, SIZE_Y)
    image_14 = transformation(aug_ids[13], image[6 * eight: 7 * eight], SIZE, SIZE_Y)

    image_15 = transformation(aug_ids[14], image[7 * eight:], SIZE, SIZE_Y)
    image_16 = transformation(aug_ids[15], image[7 * eight:], SIZE, SIZE_Y)

    # save_images(image_1, aug_ids[0])
    # save_images(image_2, aug_ids[1])
    # save_images(image_3, aug_ids[2])
    # save_images(image_4, aug_ids[3])
    # save_images(image_5, aug_ids[4])
    # save_images(image_6, aug_ids[5])
    # save_images(image_7, aug_ids[6])
    # save_images(image_8, aug_ids[7])
    # save_images(image_9, aug_ids[8])
    # save_images(image_10, aug_ids[9])
    # save_images(image_11, aug_ids[10])
    # save_images(image_12, aug_ids[11])
    # save_images(image_13, aug_ids[12])
    # save_images(image_14, aug_ids[13])
    # save_images(image_15, aug_ids[14])
    # save_images(image_16, aug_ids[15])

    image_1 = torch.cat([image_1, image_3, image_5, image_7, image_9, image_11, image_13, image_15], dim=0)
    image_2 = torch.cat([image_2, image_4, image_6, image_8, image_10, image_12, image_14, image_16], dim=0)

    # save_images(image_1, 20)
    # save_images(image_2, 21)

    _, logit_a, a = encoder(image_1.to('cuda'))
    _, logit_b, b = encoder(image_2.to('cuda'))

    total_loss, total_mean = entropy_minmax_loss(a, b, total_mean)

    if train:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return a, b, total_loss, total_mean


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def measure_acc_augments(X_test, encoder, targets):
    size = 500
    runs = len(X_test)//size
    avg_loss = 0

    print_dict = {}
    virtual_clusters = {}
    for i in range(CLASSES):
        print_dict[i] = []
        virtual_clusters[i] = []

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)

        images = X_test[test_ids, :]

        _, _, p = encoder(images.to('cuda'))

        for i in range(p.shape[0]):
            val, index = torch.max(p[i], 0)
            verdict = int(index.data.cpu().numpy())

            label = targets[test_ids[i]]

            print_dict[label].append(verdict)
            virtual_clusters[verdict].append(label)

    total_miss = 0
    clusters = set()
    for element in print_dict.keys():
        length = len(print_dict[element])
        misses = miss_classifications(print_dict[element])
        total_miss += misses

        mfe = most_frequent(print_dict[element])
        clusters.add(mfe)
        print("cluster: ",
              labels_to_imags[element],
              ", most frequent: ",
              mfe,
              ", miss-classifications: ",
              misses,
              ", miss percentage: ",
              misses / length)

    total_miss_percentage = total_miss / (runs * size)

    print()
    print("AUGMENTS avg loss: ", avg_loss / runs,
          " miss: ", total_miss,
          " data: ", runs * size,
          " miss percent: ", total_miss_percentage)
    print("Clusters found: " + str(len(clusters)) + " " + str(clusters))
    print()

    print(virtual_clusters)
    total_miss_virtual = 0
    for element in virtual_clusters.keys():
        if len(virtual_clusters[element]) == 0:
            continue
        virtual_length = len(virtual_clusters[element])

        virtual_misses = miss_classifications(virtual_clusters[element])
        total_miss_virtual += virtual_misses

        mfe = most_frequent(virtual_clusters[element])

        print("cluster: ",
              element,
              ", most frequent: ",
              labels_to_imags[mfe],
              ", miss-classifications: ",
              virtual_misses,
              ", miss percentage: ",
              virtual_misses / virtual_length)

    miss_virtual_percentage = total_miss_virtual / (runs * size)
    print("miss virtual percentage: ", miss_virtual_percentage)

    return total_miss_percentage, len(clusters), miss_virtual_percentage


def miss_classifications(cluster):
    mfe = most_frequent(cluster)
    missclassifications = 0
    for j in cluster:
        if j != mfe:
            missclassifications += 1

    return missclassifications


def most_frequent(List):
    return max(set(List), key=List.count)


def print_params(model):
    for param in model.parameters():
        print(param.data)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def train():
    with open('data\\train', 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')

    meta = unpickle('data\\meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]

    train = unpickle('data\\train')

    filenames = [t.decode('utf8') for t in train[b'filenames']]
    train_fine_labels = train[b'fine_labels']
    train_data = train[b'data']

    test = unpickle('data\\test')

    filenames = [t.decode('utf8') for t in test[b'filenames']]
    targets = test[b'fine_labels']
    test_data = test[b'data']

    X_train = list()
    for d in train_data:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
        image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
        image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
        X_train.append(image)

    X_train = np.array(X_train)
    X_train = preproccess_cifar(X_train)

    print("train shape", X_train.shape)

    X_test = list()
    for d in test_data:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
        image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
        image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
        X_test.append(image)

    X_test = np.array(X_test)
    X_test = preproccess_cifar(X_test)
    print("test shape", X_test.shape)
    targets = np.array(targets)
    print("targets shape", targets.shape)

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'cifar100_models\\actual_best' + '.model'
    actual_best_path = os.path.join(script_directory, filepath)

    filepath = 'cifar100_models\\virtual_best' + '.model'
    virtual_best_path = os.path.join(script_directory, filepath)

    encoder = OneHotNet(3, CLASSES).to('cuda')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    min_miss_percentage = 100
    max_loss_iter = 0
    most_clusters = 0
    most_clusters_iter = 0

    best_virtual_iter = 0
    min_virtual_miss_percentage = 1000

    print(encoder)
    print("X_train: ", X_train.shape, " X_test: ", X_test.shape, " targets: ", targets.shape)
    total_mean = torch.ones(CLASSES) * (1 / CLASSES)

    # test_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    # for idx, i in enumerate(targets):
    #     test_dict[i].append(idx)

    avg_loss = 0
    for iteration in range(MAX_STEPS_DEFAULT):
        encoder.train()

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        probs10, probs10_b, total_loss, total_mean = forward_block(X_train, ids, encoder, optimizer, train, total_mean)
        avg_loss += total_loss.item()

        if iteration % EVAL_FREQ_DEFAULT == 0:
            print("==================================================================================")
            print("example prediction: ", probs10[0])
            print("example prediction: ", probs10_b[0])
            print("train avg loss : ", avg_loss / BATCH_SIZE_DEFAULT)
            avg_loss = 0
            encoder.eval()
            print("total mean: ", total_mean)

            print("ITERATION: ", iteration,
                  ",  batch size: ", BATCH_SIZE_DEFAULT,
                  ",  lr: ", LEARNING_RATE_DEFAULT,
                  ",  best loss iter: ", max_loss_iter,
                  "-", min_miss_percentage,
                  ",  actual best iter: ", most_clusters_iter,
                  "-", most_clusters,
                  ",  virtual best iter: ", best_virtual_iter,
                  ",", DESCRIPTION)

            # test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            # probs10, probs10_b, total_loss, total_mean = forward_block(X_test, test_ids, encoder, optimizer, False, total_mean)

            miss_percentage, clusters, virtual_percentage = measure_acc_augments(X_test, encoder, targets)

            if virtual_percentage < min_virtual_miss_percentage:
                best_virtual_iter = iteration
                min_virtual_miss_percentage = virtual_percentage
                print("models saved virtual iter: " + str(iteration))
                torch.save(encoder, virtual_best_path)

            if clusters >= most_clusters:
                min_miss_percentage = 1 - cluster_accuracies[clusters]
                most_clusters = clusters

                if min_miss_percentage > miss_percentage:
                    cluster_accuracies[clusters] = 1 - miss_percentage
                    max_loss_iter = iteration
                    most_clusters_iter = iteration

                    print("models saved iter: " + str(iteration))
                    torch.save(encoder, actual_best_path)


def to_tensor(X):
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


def print_info(p1, p2, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}
    #image_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    image_dict = {x: [] for x in range(CLASSES[0])}
    numbers_classes_dict = {x: [] for x in range(CLASSES[0])}

    for i in range(len(test_ids)):
        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)

        mean = (p1[i] + p2[i])/2
        _, mean_index = torch.max(mean, 0)
        verdict = int(mean_index.data.cpu().numpy())

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + ", "

        label = targets[test_ids[i]]
        if label == 10:
            label = 0

        numbers_classes_dict[verdict].append(i)
        image_dict[verdict].append(labels_to_imags[label])
        print_dict[label] += string

    for i in print_dict.keys():
        print(labels_to_imags[i], " : ", print_dict[i])

    print("")
    for i in image_dict.keys():
        print(i, " : ", image_dict[i])

    return image_dict, numbers_classes_dict


def main():
    """
    Main function
    """
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')

    FLAGS, unparsed = parser.parse_known_args()

    main()