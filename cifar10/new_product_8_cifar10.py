from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import cifar10_utils
import argparse
import os

from one_hot_net import OneHotNet

from image_utils import *
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
LEARNING_RATE_DEFAULT = 4e-4

MAX_STEPS_DEFAULT = 48750

BATCH_SIZE_DEFAULT = 32

#INPUT_NET = 3072
#INPUT_NET = 5120
SIZE = 32
SIZE_Y = 32
NETS = 1

CLASSES = 10
DESCRIPTION = " Image size: " + str(SIZE) + " , Classes: " + str(CLASSES)
EPOCHS = 300

EVAL_FREQ_DEFAULT = 100
MIN_CLUSTERS_TO_SAVE = 100
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None

expected = BATCH_SIZE_DEFAULT / CLASSES

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


# def normalized_product_loss(a, b, total_mean):
#     prod = a * b
#     sum_preds = (a + b) / 2
#
#     prod_mean = prod.mean(dim=0)
#     sum_mean = sum_preds.mean(dim=0)
#
#     log = - torch.log(prod_mean / sum_mean)
#
#     scalar = log.mean()
#
#     return scalar, sum_mean
#

def product_agreement_loss(a, b, c, d):
    product = a * b * c * d
    mean = product.mean(dim=0)

    log = - torch.log(mean)

    scalar = log.mean()

    return scalar


# def another_p(a, b, c, d, mean):
#     penalty = (a.sum(dim=0) + b.sum(dim=0) + c.sum(dim=0) + d.sum(dim=0)) / 4
#
#     agreement = a * b * c * d
#     penalized_agreement = - (a+b)/2 * torch.log(agreement/penalty) * (mean.detach() / expected)
#     penalized_agreement = penalized_agreement.sum(dim=1)
#
#     scalar = penalized_agreement.mean()
#
#     return scalar, penalty
#

def penalized_product_mean(a, b, penalty, mean):
    agreement = a * b
    penalized_agreement = agreement / penalty
    penalized_agreement = penalized_agreement.sum(dim=1)

    result = - torch.log(penalized_agreement)

    ####

    normalize = (a + b)/2 * (mean.detach() / expected)
    normalize = normalize.sum(dim=1)

    result = result * normalize

    ####

    scalar = result.mean()

    return scalar


def penalized_product(a, b, penalty):
    agreement = a * b
    penalized_agreement = agreement / penalty
    penalized_agreement = penalized_agreement.sum(dim=1)

    result = - torch.log(penalized_agreement)

    scalar = result.mean()

    return scalar


def make_transformations(image, aug_ids, iter):

    fourth = image.shape[0] // 4

    image_1_a = transformation(aug_ids[0], image[0:fourth], SIZE, SIZE_Y)
    image_1_b = transformation(aug_ids[1], image[0:fourth], SIZE, SIZE_Y)
    image_1_c = transformation(aug_ids[2], image[0:fourth], SIZE, SIZE_Y)
    image_1_d = transformation(aug_ids[3], image[0:fourth], SIZE, SIZE_Y)
    image_1_e = transformation(aug_ids[4], image[0:fourth], SIZE, SIZE_Y)
    image_1_f = transformation(aug_ids[5], image[0:fourth], SIZE, SIZE_Y)
    image_1_g = transformation(aug_ids[6], image[0:fourth], SIZE, SIZE_Y)
    image_1_h = transformation(aug_ids[7], image[0:fourth], SIZE, SIZE_Y)

    image_2_a = transformation(aug_ids[8], image[fourth: 2 * fourth], SIZE, SIZE_Y)
    image_2_b = transformation(aug_ids[9], image[fourth: 2 * fourth], SIZE, SIZE_Y)
    image_2_c = transformation(aug_ids[10], image[fourth: 2 * fourth], SIZE, SIZE_Y)
    image_2_d = transformation(aug_ids[11], image[fourth: 2 * fourth], SIZE, SIZE_Y)
    image_2_e = transformation(aug_ids[12], image[fourth: 2 * fourth], SIZE, SIZE_Y)
    image_2_f = transformation(aug_ids[13], image[fourth: 2 * fourth], SIZE, SIZE_Y)
    image_2_g = transformation(aug_ids[14], image[fourth: 2 * fourth], SIZE, SIZE_Y)
    image_2_h = transformation(aug_ids[15], image[fourth: 2 * fourth], SIZE, SIZE_Y)

    image_3_a = transformation(aug_ids[16], image[2 * fourth:3 * fourth], SIZE, SIZE_Y)
    image_3_b = transformation(aug_ids[17], image[2 * fourth:3 * fourth], SIZE, SIZE_Y)
    image_3_c = transformation(aug_ids[18], image[2 * fourth:3 * fourth], SIZE, SIZE_Y)
    image_3_d = transformation(aug_ids[0], image[2 * fourth:3 * fourth], SIZE, SIZE_Y)
    image_3_e = transformation(aug_ids[8], image[2 * fourth:3 * fourth], SIZE, SIZE_Y)
    image_3_f = transformation(aug_ids[1], image[2 * fourth:3 * fourth], SIZE, SIZE_Y)
    image_3_g = transformation(aug_ids[9], image[2 * fourth:3 * fourth], SIZE, SIZE_Y)
    image_3_h = transformation(aug_ids[2], image[2 * fourth:3 * fourth], SIZE, SIZE_Y)

    image_4_a = transformation(aug_ids[10], image[3 * fourth:], SIZE, SIZE_Y)
    image_4_b = transformation(aug_ids[3], image[3 * fourth:], SIZE, SIZE_Y)
    image_4_c = transformation(aug_ids[11], image[3 * fourth:], SIZE, SIZE_Y)
    image_4_d = transformation(aug_ids[4], image[3 * fourth:], SIZE, SIZE_Y)
    image_4_e = transformation(aug_ids[12], image[3 * fourth:], SIZE, SIZE_Y)
    image_4_f = transformation(aug_ids[5], image[3 * fourth:], SIZE, SIZE_Y)
    image_4_g = transformation(aug_ids[13], image[3 * fourth:], SIZE, SIZE_Y)
    image_4_h = transformation(aug_ids[16], image[3 * fourth:], SIZE, SIZE_Y)



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

    image_1 = torch.cat([image_1_a, image_2_a, image_3_a, image_4_a], dim=0)
    image_2 = torch.cat([image_1_b, image_2_b, image_3_b, image_4_b], dim=0)
    image_3 = torch.cat([image_1_c, image_2_c, image_3_c, image_4_c], dim=0)
    image_4 = torch.cat([image_1_d, image_2_d, image_3_d, image_4_d], dim=0)

    image_5 = torch.cat([image_1_e, image_2_e, image_3_e, image_4_e], dim=0)
    image_6 = torch.cat([image_1_f, image_2_f, image_3_f, image_4_f], dim=0)
    image_7 = torch.cat([image_1_g, image_2_g, image_3_g, image_4_g], dim=0)
    image_8 = torch.cat([image_1_h, image_2_h, image_3_h, image_4_h], dim=0)

    if random.uniform(0, 1) > 0.9999:
        save_images(image_1, 19, iter)
        save_images(image_2, 20, iter)
        save_images(image_3, 21, iter)
        save_images(image_4, 22, iter)

    return image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8


def forward_block(X, ids, encoder, optimizer, train, total_mean, iter):
    number_transforms = 19
    aug_ids = np.random.choice(number_transforms, size=number_transforms, replace=False)

    image = X[ids, :]
    image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8 = make_transformations(image, aug_ids, iter)

    _, logit_a, a = encoder(image_1.to('cuda'))
    _, logit_b, b = encoder(image_2.to('cuda'))
    _, logit_c, c = encoder(image_3.to('cuda'))
    _, logit_d, d = encoder(image_4.to('cuda'))

    _, logit_e, e = encoder(image_5.to('cuda'))
    _, logit_f, f = encoder(image_6.to('cuda'))
    _, logit_g, g = encoder(image_7.to('cuda'))
    _, logit_h, h = encoder(image_8.to('cuda'))

    penalty = (a.sum(dim=0) + b.sum(dim=0) + c.sum(dim=0) + d.sum(dim=0) + e.sum(dim=0) + f.sum(dim=0) + g.sum(dim=0) + h.sum(dim=0)) / 8

    loss1 = penalized_product(a, b, penalty)
    loss2 = penalized_product(a, c, penalty)
    loss3 = penalized_product(a, d, penalty)
    loss4 = penalized_product(a, e, penalty)
    loss5 = penalized_product(a, f, penalty)
    loss6 = penalized_product(a, g, penalty)
    loss7 = penalized_product(a, h, penalty)

    loss8 = penalized_product(b, c, penalty)
    loss9 = penalized_product(b, d, penalty)
    loss10 = penalized_product(b, e, penalty)
    loss11 = penalized_product(b, f, penalty)
    loss12 = penalized_product(b, g, penalty)
    loss13 = penalized_product(b, h, penalty)

    loss14 = penalized_product(c, d, penalty)
    loss15 = penalized_product(c, e, penalty)
    loss16 = penalized_product(c, f, penalty)
    loss17 = penalized_product(c, g, penalty)
    loss18 = penalized_product(c, h, penalty)

    loss19 = penalized_product(d, e, penalty)
    loss20 = penalized_product(d, f, penalty)
    loss21 = penalized_product(d, g, penalty)
    loss22 = penalized_product(d, h, penalty)

    loss23 = penalized_product(e, f, penalty)
    loss24 = penalized_product(e, g, penalty)
    loss25 = penalized_product(e, h, penalty)

    loss26 = penalized_product(f, g, penalty)
    loss27 = penalized_product(f, h, penalty)

    loss28 = penalized_product(g, h, penalty)

    total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10+ loss11+ loss12+ loss13+ loss14 + loss15+loss16 +  loss17+loss18 +loss19+loss20+loss21+loss22+loss23+loss24+loss25+loss26+loss27+loss28

    if train:

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return a, b, total_loss, total_mean


def save_cluster(original_image, cluster, transf, iter):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    #sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"iter_{iter}_transf_{transf}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def measure_test_loss(x_test, encoder, total_mean):
    size = 500
    runs = len(x_test) // size
    sum_loss = 0

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)

        with torch.no_grad():
            test_preds_1, test_preds_2, test_total_loss, total_mean = forward_block(x_test, test_ids, encoder, [], False,
                                                                       total_mean, 0)

        sum_loss += test_total_loss.item()

    avg_loss = sum_loss / runs

    print()
    print("Avg test loss: ", avg_loss)
    print()

    return avg_loss


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
    print("Clusters found: " + str(len(clusters)))
    print()

    #print(virtual_clusters)
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
    print("acc virtual percentage: ", 1-miss_virtual_percentage)

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
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = cifar10_utils.load_cifar10(cifar10_utils.CIFAR10_FOLDER)
    X_train, y_train, X_test, targets = cifar10_utils.preprocess_cifar10_data(X_train_raw, y_train_raw, X_test_raw,
                                                                              y_test_raw)

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)

    X_train /= 255
    X_test /= 255

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = f'PPA_8_{BATCH_SIZE_DEFAULT}_lr{LEARNING_RATE_DEFAULT}'
    virtual_best_path = os.path.join(script_directory, filepath)

    # epoch 158.
    #encoder = torch.load("PPA_4_128_lr2_1.model")

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
    total_mean = torch.ones(CLASSES) * expected
    total_mean = total_mean.cuda()

    # test_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    # for idx, i in enumerate(targets):
    #     test_dict[i].append(idx)

    total_iters = 0
    avg_loss = 0
    for epoch in range(EPOCHS):
        print("Epoch: ", epoch)
        ids = np.random.choice(len(X_train), size=len(X_train), replace=False)

        runs = len(X_train) // BATCH_SIZE_DEFAULT

        iteration = 0

        for j in range(runs):
            current_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
            encoder.train()
            iter_ids = ids[current_ids]

            train = True
            probs10, probs10_b, total_loss, total_mean = forward_block(X_train, iter_ids, encoder, optimizer, train,
                                                                       total_mean, iteration)
            avg_loss += total_loss.item()
            iteration += 1
            total_iters += 1

        print("==================================================================================")

        if iteration % 2000 == 0:
            print("example prediction: ", probs10[0])
            print("example prediction: ", probs10_b[0])
            print("total mean: ", total_mean)
            print()

        print("train avg loss : ", avg_loss / runs)
        avg_loss = 0
        encoder.eval()

        avg_test_loss = measure_test_loss(X_test, encoder, total_mean)
        miss_percentage, clusters, virtual_percentage = measure_acc_augments(X_test, encoder, targets)

        #torch.save(encoder, virtual_best_path+"_last")

        if avg_test_loss < min_virtual_miss_percentage:
            best_virtual_iter = total_iters
            min_virtual_miss_percentage = avg_test_loss
            print("models saved virtual iter: " + str(total_iters))
            torch.save(encoder, virtual_best_path + "_" + str(epoch//100) + '.model')

        print("ITERATION: ", total_iters,
              ",  batch size: ", BATCH_SIZE_DEFAULT,
              ",  lr: ", LEARNING_RATE_DEFAULT,
              ",  best loss iter: ", max_loss_iter,
              "-", min_miss_percentage,
              ",  actual best iter: ", most_clusters_iter, "-", most_clusters,
              ",  virtual best iter: ", best_virtual_iter, " - ", min_virtual_miss_percentage,
              ",", DESCRIPTION)

        # if clusters >= most_clusters:
        #     min_miss_percentage = 1 - cluster_accuracies[clusters]
        #     most_clusters = clusters
        #
        #     if min_miss_percentage > miss_percentage:
        #         cluster_accuracies[clusters] = 1 - miss_percentage
        #         max_loss_iter = total_iters
        #         most_clusters_iter = total_iters
        #
        #         print("models actual_best_path iter: " + str(total_iters))


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