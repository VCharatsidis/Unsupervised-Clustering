from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.utils.prune as prune

import sys

import argparse
import os

from mixed_net import Mixed

from image_utils import *
import random

from torchvision.utils import make_grid
import matplotlib
import pickle

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(sci_mode=False)

EPS = sys.float_info.epsilon

#EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4

MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 128

EMBEDINGS = 2048
SIZE = 32
SIZE_Y = 32
NETS = 1

QUEUE = 10
EPOCHS = 1000
TR = 4

CLASSES = 100
DESCRIPTION = " Image size: " + str(SIZE) + " , Classes: " + str(CLASSES)

EVAL_FREQ_DEFAULT = 200
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None

square = torch.ones(BATCH_SIZE_DEFAULT, BATCH_SIZE_DEFAULT)
ZERO_DIAG = square.fill_diagonal_(0)
first_part = torch.cat([ZERO_DIAG, ZERO_DIAG, ZERO_DIAG, ZERO_DIAG], dim=1)
adj_matrix = torch.cat([first_part, first_part, first_part, first_part], dim=0)
adj_matrix = adj_matrix.cuda()

big_diag = torch.ones(TR * BATCH_SIZE_DEFAULT, TR * BATCH_SIZE_DEFAULT)
ZERO_BIG_DIAG = big_diag.fill_diagonal_(0).cuda()


first = True

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
                       8: "randcom_crop_upscale_gauss_blur",
                       9: "randcom_crop_upscale_sobel_total",
                       10: "randcom_crop_upscale_rev_pixels",
                       11: "no_jitter_rotate(image, -46)",
                       12: "randcom_crop_upscale(image, 20",
                       13: "randcom_crop_upscale(image, 22",
                       14: "randcom_crop_upscale(image, 26",
                       15: "no_jitter_random_corpse(image, 22",
                       16: "randcom_crop_upscale(image, 18",
                       17: "random_erase",
                       18: "noise",
                       19: "image_1",
                       20: "image_2",
                       21: "image_3",
                       22: "image_4"}


labels_to_imags = {}
for i in range(CLASSES):
    labels_to_imags[i] = i


def save_images(images, transformation, iter):
    print(transformations_dict[transformation], " ", transformation)
    numpy_cluster = images.cpu().detach()
    save_cluster(numpy_cluster, transformations_dict[transformation], transformation, iter)


def new_agreement(product, denominator, rev_prod, class_adj_matrix, isCurrent):
    prod_2 = 1 - rev_prod
    attraction = torch.mm(product, prod_2.transpose(0, 1))
    repel = torch.mm(product, rev_prod.transpose(0, 1))

    denominator = denominator.unsqueeze(dim=1)

    attraction = attraction / denominator
    repel = repel / denominator

    # print(repel.shape)
    # print(attraction.shape)
    # print(class_adj_matrix.shape)
    # print()

    if isCurrent:

        class_adj_matrix = class_adj_matrix * adj_matrix + (1 - adj_matrix)
        total_matrix = repel * (1 - class_adj_matrix) + attraction * class_adj_matrix

        log_total = - torch.log(total_matrix)
        diagonal_elements_bonus = 25 * (1 - adj_matrix) * log_total
        total = log_total + diagonal_elements_bonus
        total = total.mean()

        return total

    else:
        total_matrix = repel * (1 - class_adj_matrix) + attraction * class_adj_matrix

    #total_matrix[(total_matrix < EPS).data] = EPS

    log_total = - torch.log(total_matrix)
    # diagonal_elements_bonus = 0.5 * BATCH_SIZE_DEFAULT * class_adj_matrix * log_total
    total = log_total  #+ diagonal_elements_bonus
    total = total.mean()

    return total


def queue_agreement(product, denominator, rev_prod):
    transposed = rev_prod.transpose(0, 1)

    nondiag = torch.mm(product, transposed)
    nondiag = nondiag / denominator.unsqueeze(dim=1)

    log_nondiag = - torch.log(nondiag)
    negative = log_nondiag.mean()

    return negative


def make_transformations(image, aug_ids):

    eight = image.shape[0] // 8

    image_1_a = transformation(aug_ids[0], image[0:eight], SIZE, SIZE_Y)
    image_1_b = transformation(aug_ids[1], image[0:eight], SIZE, SIZE_Y)
    image_1_c = transformation(aug_ids[2], image[0:eight], SIZE, SIZE_Y)
    image_1_d = transformation(aug_ids[16], image[0:eight], SIZE, SIZE_Y)

    image_2_a = transformation(aug_ids[3], image[eight: 2 * eight], SIZE, SIZE_Y)
    image_2_b = transformation(aug_ids[4], image[eight: 2 * eight], SIZE, SIZE_Y)
    image_2_c = transformation(aug_ids[5], image[eight: 2 * eight], SIZE, SIZE_Y)
    image_2_d = transformation(aug_ids[6], image[eight: 2 * eight], SIZE, SIZE_Y)

    image_3_a = transformation(aug_ids[6], image[2 * eight: 3 * eight], SIZE, SIZE_Y)
    image_3_b = transformation(aug_ids[7], image[2 * eight: 3 * eight], SIZE, SIZE_Y)
    image_3_c = transformation(aug_ids[8], image[2 * eight: 3 * eight], SIZE, SIZE_Y)
    image_3_d = transformation(aug_ids[10], image[2 * eight: 3 * eight], SIZE, SIZE_Y)

    image_4_a = transformation(aug_ids[9], image[3 * eight: 4 * eight], SIZE, SIZE_Y)
    image_4_b = transformation(aug_ids[10], image[3 * eight: 4 * eight], SIZE, SIZE_Y)
    image_4_c = transformation(aug_ids[11], image[3 * eight: 4 * eight], SIZE, SIZE_Y)
    image_4_d = transformation(aug_ids[16], image[3 * eight: 4 * eight], SIZE, SIZE_Y)

    image_5_a = transformation(aug_ids[12], image[4 * eight: 5 * eight], SIZE, SIZE_Y)
    image_5_b = transformation(aug_ids[13], image[4 * eight: 5 * eight], SIZE, SIZE_Y)
    image_5_c = transformation(aug_ids[14], image[4 * eight: 5 * eight], SIZE, SIZE_Y)
    image_5_d = transformation(aug_ids[2], image[4 * eight: 5 * eight], SIZE, SIZE_Y)

    image_6_a = transformation(aug_ids[15], image[5 * eight: 6 * eight], SIZE, SIZE_Y)
    image_6_b = transformation(aug_ids[0], image[5 * eight: 6 * eight], SIZE, SIZE_Y)
    image_6_c = transformation(aug_ids[3], image[5 * eight: 6 * eight], SIZE, SIZE_Y)
    image_6_d = transformation(aug_ids[7], image[5 * eight: 6 * eight], SIZE, SIZE_Y)

    image_7_a = transformation(aug_ids[4], image[6 * eight: 7 * eight], SIZE, SIZE_Y)
    image_7_b = transformation(aug_ids[8], image[6 * eight: 7 * eight], SIZE, SIZE_Y)
    image_7_c = transformation(aug_ids[9], image[6 * eight: 7 * eight], SIZE, SIZE_Y)
    image_7_d = transformation(aug_ids[1], image[6 * eight: 7 * eight], SIZE, SIZE_Y)

    image_8_a = transformation(aug_ids[5], image[7 * eight:], SIZE, SIZE_Y)
    image_8_b = transformation(aug_ids[11], image[7 * eight:], SIZE, SIZE_Y)
    image_8_c = transformation(aug_ids[18], image[7 * eight:], SIZE, SIZE_Y)
    image_8_d = transformation(aug_ids[17], image[7 * eight:], SIZE, SIZE_Y)

    # save_images(image_1_a, aug_ids[0], iter)
    # save_images(image_1_b, aug_ids[1], iter)
    # save_images(image_1_c, aug_ids[2], iter)
    #
    # save_images(image_2_a, aug_ids[3], iter)
    # save_images(image_2_b, aug_ids[4], iter)
    # save_images(image_2_c, aug_ids[5], iter)
    # save_images(image_2_d, aug_ids[6], iter)
    #
    # save_images(image_3_b, aug_ids[7], iter)
    # save_images(image_3_c, aug_ids[8], iter)
    # save_images(image_3_d, aug_ids[9], iter)
    #
    # save_images(image_4_a, aug_ids[10], iter)
    # save_images(image_4_c, aug_ids[11], iter)
    #
    # save_images(image_5_a, aug_ids[12], iter)
    # save_images(image_5_b, aug_ids[13], iter)
    # save_images(image_5_c, aug_ids[14], iter)
    #
    # save_images(image_6_a, aug_ids[15], iter)
    #
    # save_images(image_4_d, aug_ids[16], iter)
    # save_images(image_8_d, aug_ids[17], iter)
    #save_images(image_8_c, 18, iter)

    image_1 = torch.cat([image_1_a, image_2_a, image_3_a, image_4_a, image_5_a, image_6_a, image_7_a, image_8_a], dim=0)
    image_2 = torch.cat([image_1_b, image_2_b, image_3_b, image_4_b, image_5_b, image_6_b, image_7_b, image_8_b], dim=0)
    image_3 = torch.cat([image_1_c, image_2_c, image_3_c, image_4_c, image_5_c, image_6_c, image_7_c, image_8_c], dim=0)
    image_4 = torch.cat([image_1_d, image_2_d, image_3_d, image_4_d, image_5_d, image_6_d, image_7_d, image_8_d], dim=0)

    if random.uniform(0, 1) > 0.9999:
        iter = random.randint(0, 10000)
        save_images(image_1, 19, iter)
        save_images(image_2, 20, iter)
        save_images(image_3, 21, iter)
        save_images(image_4, 22, iter)

    return image_1, image_2, image_3, image_4


def forward_block(X, ids, encoder, optimizer, train, rev_product, moving_mean, old_product_labels, old_penalty_labels):
    global first
    number_transforms = 19
    aug_ids = np.random.choice(number_transforms, size=number_transforms, replace=False)
    #
    # image = X[ids, :]
    #
    # eight = image.shape[0] // 8
    #
    # image_1 = transformation(aug_ids[0], image[0:eight], SIZE, SIZE_Y)
    # image_2 = transformation(aug_ids[1], image[0:eight], SIZE, SIZE_Y)
    #
    # image_3 = transformation(aug_ids[2], image[eight: 2 * eight], SIZE, SIZE_Y)
    # image_4 = transformation(aug_ids[3], image[eight: 2 * eight], SIZE, SIZE_Y)
    #
    # image_5 = transformation(aug_ids[4], image[2 * eight: 3 * eight], SIZE, SIZE_Y)
    # image_6 = transformation(aug_ids[5], image[2 * eight: 3 * eight], SIZE, SIZE_Y)
    #
    # image_7 = transformation(aug_ids[6], image[3 * eight: 4 * eight], SIZE, SIZE_Y)
    # image_8 = transformation(aug_ids[7], image[3 * eight: 4 * eight], SIZE, SIZE_Y)
    #
    # image_9 = transformation(aug_ids[8], image[4 * eight: 5 * eight], SIZE, SIZE_Y)
    # image_10 = transformation(aug_ids[9], image[4 * eight: 5 * eight], SIZE, SIZE_Y)
    #
    # image_11 = transformation(aug_ids[10], image[5 * eight: 6 * eight], SIZE, SIZE_Y)
    # image_12 = transformation(aug_ids[11], image[5 * eight: 6 * eight], SIZE, SIZE_Y)
    #
    # image_13 = transformation(aug_ids[12], image[6 * eight: 7 * eight], SIZE, SIZE_Y)
    # image_14 = transformation(aug_ids[13], image[6 * eight: 7 * eight], SIZE, SIZE_Y)
    #
    # image_15 = transformation(aug_ids[14], image[7 * eight:], SIZE, SIZE_Y)
    # image_16 = transformation(aug_ids[15], image[7 * eight:], SIZE, SIZE_Y)
    #
    # # save_images(image_1, aug_ids[0])
    # # save_images(image_2, aug_ids[1])
    # # save_images(image_3, aug_ids[2])
    # # save_images(image_4, aug_ids[3])
    # # save_images(image_5, aug_ids[4])
    # # save_images(image_6, aug_ids[5])
    # # save_images(image_7, aug_ids[6])
    # # save_images(image_8, aug_ids[7])
    # # save_images(image_9, aug_ids[8])
    # # save_images(image_10, aug_ids[9])
    # # save_images(image_11, aug_ids[10])
    # # save_images(image_12, aug_ids[11])
    # # save_images(image_13, aug_ids[12])
    # # save_images(image_14, aug_ids[13])
    # # save_images(image_15, aug_ids[14])
    # # save_images(image_16, aug_ids[15])
    #
    # image_1 = torch.cat([image_1, image_3, image_5, image_7, image_9, image_11, image_13, image_15], dim=0)
    # image_2 = torch.cat([image_2, image_4, image_6, image_8, image_10, image_12, image_14, image_16], dim=0)

    # save_images(image_1, 20)
    # save_images(image_2, 21)

    image = X[ids, :]
    image_1, image_2, image_3, image_4 = make_transformations(image, aug_ids)

    _, c_a, pc_a, a = encoder(image_1.to('cuda'))
    _, c_b, pc_b, b = encoder(image_2.to('cuda'))
    _, c_c, pc_c, c = encoder(image_3.to('cuda'))
    _, c_d, pc_d, d = encoder(image_4.to('cuda'))

    all_predictions = torch.cat([a, b, c, d], dim=0)

    current_reverse = 1 - all_predictions
    denominator = torch.cat([a.sum(dim=1), b.sum(dim=1), c.sum(dim=1), d.sum(dim=1)], dim=0)

    penalty = (pc_a.sum(dim=0) + pc_b.sum(dim=0) + pc_c.sum(dim=0) + pc_d.sum(dim=0)) / 4

    penalty_loss1, penalty_mean = penalized_product(pc_a, pc_b, penalty)
    penalty_loss2, penalty_mean = penalized_product(pc_a, pc_c, penalty)
    penalty_loss3, penalty_mean = penalized_product(pc_a, pc_d, penalty)

    penalty_loss4, penalty_mean = penalized_product(pc_b, pc_c, penalty)
    penalty_loss5, penalty_mean = penalized_product(pc_b, pc_d, penalty)

    penalty_loss6, penalty_mean = penalized_product(pc_c, pc_d, penalty)


    # classification_loss1, current_mean = penalized_product(c_a, c_b)
    # classification_loss2, current_mean = penalized_product(c_a, c_c)
    # classification_loss3, current_mean = penalized_product(c_a, c_d)
    #
    # classification_loss4, current_mean = penalized_product(c_b, c_c)
    # classification_loss5, current_mean = penalized_product(c_b, c_d)
    #
    # classification_loss6, current_mean = penalized_product(c_c, c_d)
    #
    penalty_loss = penalty_loss1 + penalty_loss2 + penalty_loss3 + penalty_loss4 + penalty_loss5 + penalty_loss6
    # classification_loss = classification_loss1 + classification_loss2 + classification_loss3 + classification_loss4+ classification_loss5 + classification_loss6


    #penalty_loss, _ = class_mean_probs_loss(pc_a, pc_b, pc_c, pc_d, moving_mean)
    classification_loss, current_mean = penalized_product_generalized(c_a, c_b, c_c, c_d)

    product_labels = torch.cat([c_a, c_b, c_c, c_d], dim=0)
    penalty_labels = torch.cat([pc_a, pc_b, pc_c, pc_d], dim=0)

    class_adj_m = class_adj_matrix(product_labels) * class_adj_matrix(penalty_labels)

    new_loss = new_agreement(all_predictions, denominator, current_reverse, class_adj_m, True)

    if first or not train:
        if first:
            print("class adj matrix mean value: ", class_adj_m.mean())

        bin_loss = new_loss
        total_loss = bin_loss + classification_loss + penalty_loss
        rev_product = current_reverse.detach()
        old_product_labels = product_labels.detach()
        old_penalty_labels = penalty_labels.detach()
        first = False

    else:
        old_adj_m = class_old_adj_matrix(product_labels, old_product_labels) * class_old_adj_matrix(penalty_labels, old_penalty_labels)

        old_loss = new_agreement(all_predictions, denominator, rev_product, old_adj_m, False)
        rev_product = torch.cat([rev_product, current_reverse.detach()])
        old_product_labels = torch.cat([old_product_labels, product_labels.detach()])
        old_penalty_labels = torch.cat([old_penalty_labels, penalty_labels.detach()])

        bin_loss = new_loss + old_loss
        total_loss = bin_loss + classification_loss + penalty_loss

    if train:
        moving_mean = 0.9 * moving_mean.cuda() + 0.1 * current_mean.detach()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return a, b, bin_loss, rev_product, classification_loss + penalty_loss, moving_mean, old_product_labels, old_penalty_labels


# def assign_labels(a, b):
#     predictions = (a + b) / 2
#
#     max_elems, indexes = predictions.max(dim=1)
#
#     # print(max_elems.shape)
#     # print(indexes.shape)
#     #
#
#     # #print(predictions[0])
#     # print(indexes[0])
#
#     return indexes, max_elems


def class_old_adj_matrix(labels, old_labels):
    transposed = old_labels.transpose(0, 1)

    class_adj_matrix = torch.mm(labels, transposed)

    return class_adj_matrix.detach()


def class_adj_matrix(all_labels):
    transposed = all_labels.transpose(0, 1)

    class_adj_matrix = torch.mm(all_labels, transposed)

    # class_adj_matrix = all_predictions * all_predictions.unsqueeze(dim=1)

    return class_adj_matrix.detach()


def penalized_product_generalized(a, b, c, d):
    penalty = (a.sum(dim=0) + b.sum(dim=0) + c.sum(dim=0) + d.sum(dim=0)) / 4

    agreement = (a * b + a * c + a * d + b * c + b * d + c * d) / 6
    penalized_agreement = agreement / penalty
    penalized_agreement = penalized_agreement.sum(dim=1)

    result = - torch.log(penalized_agreement)

    scalar = result.mean()

    return scalar, penalty


def penalized_product(a, b, penalty):

    agreement = a * b
    penalized_agreement = agreement / penalty
    penalized_agreement = penalized_agreement.sum(dim=1)

    result = - torch.log(penalized_agreement)

    scalar = result.mean()

    return scalar, penalty


def class_mean_probs_loss(a, b, c, d, moving_mean):
    product = a * b * c * d
    current_mean = product.mean(dim=0)

    interpolation = 0.4 * current_mean + 0.6 * moving_mean
    log = - torch.log(interpolation)
    scalar = log.mean()

    return scalar, current_mean


def save_cluster(original_image, cluster, transf, iter):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    #sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"iter_{iter}_transf_{transf}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def measure_acc_clustering(X_test, encoder, targets, isProduct):
    size = 200
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

        if isProduct:
            _, p, _, _ = encoder(images.to('cuda'))
        else:
            _, _, p, _ = encoder(images.to('cuda'))

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
        # print("cluster: ",
        #       labels_to_imags[element],
        #       ", most frequent: ",
        #       mfe,
        #       ", miss-classifications: ",
        #       misses,
        #       ", miss percentage: ",
        #       misses / length)

    total_miss_percentage = total_miss / (runs * size)

    print()
    print("AUGMENTS avg loss: ", avg_loss / runs,
          " miss: ", total_miss,
          " data: ", runs * size,
          " miss percent: ", total_miss_percentage)
    print("Clusters found: " + str(len(clusters)) + " " + str(clusters))
    print()

    total_miss_virtual = 0
    for element in virtual_clusters.keys():
        if len(virtual_clusters[element]) == 0:
            continue
        #virtual_length = len(virtual_clusters[element])

        virtual_misses = miss_classifications(virtual_clusters[element])
        total_miss_virtual += virtual_misses

        #mfe = most_frequent(virtual_clusters[element])

        # print("cluster: ",
        #       element,
        #       ", most frequent: ",
        #       labels_to_imags[mfe],
        #       ", miss-classifications: ",
        #       virtual_misses,
        #       ", size: ",
        #       virtual_length,
        #       ", miss percentage: ",
        #       virtual_misses / virtual_length)

    miss_virtual_percentage = total_miss_virtual / (runs * size)
    print("miss virtual percentage: ", miss_virtual_percentage)

    return total_miss_percentage, len(clusters), miss_virtual_percentage


def measure_acc_augments(x_test, encoder, rev_product, moving_mean, old_product_labels, old_penalty_labels):
    size = BATCH_SIZE_DEFAULT
    runs = len(x_test) // size
    sum_loss = 0
    sum_class_loss = 0
    print(rev_product.shape)

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)

        with torch.no_grad():
            test_preds_1, test_preds_2, test_total_loss, rev_product, class_loss, moving_mean, old_product_labels, old_penalty_labels = forward_block(x_test, test_ids, encoder, [], False, rev_product, moving_mean, old_product_labels, old_penalty_labels)

        sum_loss += test_total_loss.item()
        sum_class_loss += class_loss.item()

    avg_loss = sum_loss / runs
    avg_class_loss = sum_class_loss / runs

    print()
    print("Avg test loss: ", avg_loss)
    print("Avg test class loss: ", avg_class_loss)
    print()

    return avg_loss


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
    global first
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

    filepath = 'cifar100_models\\mixed_disentangle' + '.model'
    clusters_net_path = os.path.join(script_directory, filepath)

    encoder = Mixed(3, EMBEDINGS, CLASSES).to('cuda')

    print(encoder)

    #print(list(encoder.brain[0].weight))
    #prune.random_unstructured(encoder.brain[0], name="weight", amount=0.6)
    #print(list(encoder.brain[0].weight))

    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    min_miss_percentage = 100
    max_loss_iter = 0

    test_best_loss = 1000

    print("X_train: ", X_train.shape, " X_test: ", X_test.shape, " targets: ", targets.shape)


    moving_mean = torch.ones(CLASSES) / (CLASSES*CLASSES*CLASSES*CLASSES)
    moving_mean = moving_mean.cuda()

    avg_loss = 0
    total_iters = 0
    avg_class_loss = 0

    for epoch in range(EPOCHS):
        print("Epoch: ", epoch)
        ids = np.random.choice(len(X_train), size=len(X_train), replace=False)

        runs = len(X_train) // BATCH_SIZE_DEFAULT

        rev_product = torch.ones([BATCH_SIZE_DEFAULT, EMBEDINGS]).cuda()
        product_labels = torch.ones([BATCH_SIZE_DEFAULT, CLASSES]).cuda()
        penalty_labels = torch.ones([BATCH_SIZE_DEFAULT, CLASSES]).cuda()

        first = True
        iteration = 0

        for j in range(runs):
            current_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
            encoder.train()
            iter_ids = ids[current_ids]

            train = True
            probs10, probs10_b, total_loss, rev_product, class_loss, moving_mean, product_labels, penalty_labels = forward_block(X_train, iter_ids, encoder, optimizer, train, rev_product, moving_mean, product_labels, penalty_labels)
            avg_loss += total_loss.item()
            avg_class_loss += class_loss.item()
            iteration += 1
            total_iters += 1

            if iteration >= QUEUE:
                rev_product = rev_product[TR * BATCH_SIZE_DEFAULT:, :]
                product_labels = product_labels[TR * BATCH_SIZE_DEFAULT:, :]
                penalty_labels = penalty_labels[TR * BATCH_SIZE_DEFAULT:, :]

        print("==================================================================================")
        print("moving mean: ", moving_mean)
        print("batch mean ones: ",
              (np.where(probs10.data.cpu().numpy() > 0.5))[0].shape[0] / (probs10[0].shape[0] * BATCH_SIZE_DEFAULT))

        count_common_elements(probs10)
        print()

        print("train avg loss : ", avg_loss / runs)
        print("train avg class loss : ", avg_class_loss / runs)
        avg_loss = 0
        avg_class_loss = 0
        encoder.eval()

        test_loss = measure_acc_augments(X_test, encoder, rev_product, moving_mean, product_labels, penalty_labels)
        z = measure_acc_clustering(X_test, encoder, targets, True)
        z2 = measure_acc_clustering(X_test, encoder, targets, False)

        if test_loss < test_best_loss:
            test_best_loss = test_loss
            max_loss_iter = total_iters
            min_miss_percentage = test_loss
            print("models saved iter: " + str(total_iters))
            torch.save(encoder, clusters_net_path)

        print("EPOCH: ", epoch,
              "Total ITERATION: ", total_iters,
              " epoch iter: ", iteration,
              ",  batch size: ", BATCH_SIZE_DEFAULT,
              ",  lr: ", LEARNING_RATE_DEFAULT,
              ",  best loss iter: ", max_loss_iter,
              "-", min_miss_percentage
              , DESCRIPTION)


def count_common_elements(p):
    sum_commons = 0
    counter = 0
    p = torch.round(p)
    for i in range(p.shape[0]):
        for j in range(p.shape[0]):
            if i == j:
                continue

            product = p[i].data.cpu().numpy() * p[j].data.cpu().numpy()
            commons = np.where(product > 0.5)[0].shape[0]
            #print(commons)

            sum_commons += commons
            counter += 1

    print("Mean common elements: ", (sum_commons / EMBEDINGS) / counter)



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