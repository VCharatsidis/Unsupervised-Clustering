from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.utils.prune as prune

import sys
import scipy.io as sio
import argparse
import os

from binary_net import DeepBinBrainCifar
#from AlexNet import AlexNet

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

LEARNING_RATE_DEFAULT = 4e-4

MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 128

EMBEDINGS = 100
SIZE = 32
SIZE_Y = 32
NETS = 1

EPOCHS = 400
TR = 1

CLASSES = 100
DESCRIPTION = " Image size: " + str(SIZE) + " , Classes: " + str(CLASSES)
QUEUE = 150

EVAL_FREQ_DEFAULT = 250
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None

square = torch.ones(BATCH_SIZE_DEFAULT, BATCH_SIZE_DEFAULT)
adj_matrix = square.fill_diagonal_(0)
# first_part = torch.cat([adj_matrix, adj_matrix], dim=1)
# adj_matrix = torch.cat([first_part, first_part], dim=0)
adj_matrix = adj_matrix.cuda()

# big_diag = torch.ones(2 * BATCH_SIZE_DEFAULT, 2 * BATCH_SIZE_DEFAULT)
# ZERO_BIG_DIAG = big_diag.fill_diagonal_(0)
# ZERO_BIG_DIAG = ZERO_BIG_DIAG.cuda()

#ELEMENTS_EXCEPT_DIAG = 2 * BATCH_SIZE_DEFAULT * (BATCH_SIZE_DEFAULT - 1)

first = True

cluster_accuracies = {}
for i in range(CLASSES):
    cluster_accuracies[i] = 0


class_numbers = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 0: 0}


transformations_dict = {0: "original 0",
                       1: "scale 1",
                       2: "rotate 2",
                       3: "reverse pixel value 3",
                       4: "sobel total 4",
                       5: "sobel x 5",
                       6: "sobel y 6",
                       7: "gaussian blur 7",
                       8: "randcom_crop_upscale_gauss_blur 8",
                       9: "randcom_crop_upscale sobel 9",
                       10: "random crop reverse pixel 10",
                       11: "random crop rotate 11",
                       12: "random crop soble rotate 12",
                       13: "randcom_crop_upscale 13",
                       14: "randcom_crop_upscale 14",
                       15:"randcom_crop_upscale sobel y 15",
                       16: " randcom crop 18x18 16"}


labels_to_imags = {}
for i in range(CLASSES):
    labels_to_imags[i] = i


def save_images(images, transformation):
    print(transformations_dict[transformation])
    numpy_cluster = images.cpu().detach()
    save_cluster(numpy_cluster, transformations_dict[transformation], 0)


def new_agreement(product, denominator, rev_prod, class_adj_matrix):
    prod_2 = 1 - rev_prod
    attraction = (torch.mm(product, prod_2.transpose(0, 1)) + EPS) * (1 - class_adj_matrix)
    repel = (torch.mm(product, rev_prod.transpose(0, 1)) + EPS) * class_adj_matrix

    attraction = attraction / denominator
    repel = repel / denominator

    log_total = - torch.log(attraction + repel)

    attraction_bonus = 30 * (1 - class_adj_matrix) * log_total

    total = log_total + attraction_bonus

    return total.mean()


def class_old_adj_matrix(labels, old_labels):
    square = torch.ones(BATCH_SIZE_DEFAULT, len(old_labels)).cuda()

    square[(labels.unsqueeze(dim=1) == old_labels).data] = 0
    # print(labels)
    # print(old_labels)
    # print(square)
    # input()
    #class_adj_matrix = torch.cat([square, square], dim=0)

    return square


def class_adj_matrix(labels):
    square = torch.ones(BATCH_SIZE_DEFAULT, BATCH_SIZE_DEFAULT).cuda()

    labels_vertical = labels.unsqueeze(dim=1)
    square[(labels == labels_vertical).data] = 0
    #
    # print(labels)
    # print(square)
    # input()
    # first_part = torch.cat([square, square], dim=1)
    # class_adj_matrix = torch.cat([first_part, first_part], dim=0)

    return square


def forward_block(X, ids, encoder, optimizer, train, rev_product, y, old_ids):
    global first
    image = X[ids, :]

    # number_transforms = 19
    # aug_ids = np.random.choice(number_transforms, size=number_transforms, replace=False)
    #
    # eight = image.shape[0] // 8
    #
    # #image_1 = transformation(aug_ids[0], image[0:eight], SIZE, SIZE_Y)
    # image_2 = transformation(aug_ids[1], image[0:eight], SIZE, SIZE_Y)
    #
    # #image_3 = transformation(aug_ids[2], image[eight: 2 * eight], SIZE, SIZE_Y)
    # image_4 = transformation(aug_ids[3], image[eight: 2 * eight], SIZE, SIZE_Y)
    #
    # #image_5 = transformation(aug_ids[4], image[2 * eight: 3 * eight], SIZE, SIZE_Y)
    # image_6 = transformation(aug_ids[5], image[2 * eight: 3 * eight], SIZE, SIZE_Y)
    #
    # #image_7 = transformation(aug_ids[6], image[3 * eight: 4 * eight], SIZE, SIZE_Y)
    # image_8 = transformation(aug_ids[7], image[3 * eight: 4 * eight], SIZE, SIZE_Y)
    #
    # #image_9 = transformation(aug_ids[8], image[4 * eight: 5 * eight], SIZE, SIZE_Y)
    # image_10 = transformation(aug_ids[9], image[4 * eight: 5 * eight], SIZE, SIZE_Y)
    #
    # #image_11 = transformation(aug_ids[10], image[5 * eight: 6 * eight], SIZE, SIZE_Y)
    # image_12 = transformation(aug_ids[11], image[5 * eight: 6 * eight], SIZE, SIZE_Y)
    #
    # #image_13 = transformation(aug_ids[12], image[6 * eight: 7 * eight], SIZE, SIZE_Y)
    # image_14 = transformation(aug_ids[13], image[6 * eight: 7 * eight], SIZE, SIZE_Y)
    #
    # #image_15 = transformation(aug_ids[14], image[7 * eight:], SIZE, SIZE_Y)
    # image_16 = transformation(aug_ids[15], image[7 * eight:], SIZE, SIZE_Y)

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

    #image_1 = image
    #image_1 = torch.cat([image_1, image_3, image_5, image_7, image_9, image_11, image_13, image_15], dim=0)
    #image_2 = torch.cat([image_2, image_4, image_6, image_8, image_10, image_12, image_14, image_16], dim=0)

    # save_images(image_1, 20)
    # save_images(image_2, 21)

    _, _, a = encoder(image.to('cuda'))
    #_, _, b = encoder(image_2.to('cuda'))

    #all_predictions = torch.cat([a, b], dim=0)
    all_predictions = a

    current_reverse = 1 - all_predictions
    # denominator = torch.cat([a.sum(dim=1), b.sum(dim=1)], dim=0)
    denominator = a.sum(dim=1)
    denominator = denominator.unsqueeze(dim=1) + 1

    class_adj_m = class_adj_matrix(y[ids])
    new_loss = new_agreement(all_predictions, denominator, current_reverse, class_adj_m)

    if first or not train:
        # print("first: ", first)
        total_loss = new_loss
        rev_product = current_reverse.detach()
        first = False

    else:
        # print("queue_agreement")
        old_adj_m = class_old_adj_matrix(y[ids], y[old_ids])

        old_loss = new_agreement(all_predictions, denominator, rev_product, old_adj_m)
        rev_product = torch.cat([rev_product, current_reverse.detach()])
        total_loss = new_loss + old_loss

    if train:
        for z in range(TR):
            old_ids.extend(ids)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return a, b, total_loss, rev_product, old_ids


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def measure_acc_augments(x_test, encoder, rev_product, targets):
    size = BATCH_SIZE_DEFAULT
    runs = len(x_test) // size
    sum_loss = 0

    print(rev_product.shape)

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)

        with torch.no_grad():
            test_preds_1, test_preds_2, test_total_loss, rev_product, old_ids = forward_block(x_test, test_ids, encoder, [], False, rev_product, targets, [])

        sum_loss += test_total_loss.item()

    avg_loss = sum_loss / runs

    print()
    print("Avg test loss: ", avg_loss)
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
    #unsupervised_data = sio.loadmat('data\\extra_32x32.mat')
    train_data = sio.loadmat('data\\train_32x32.mat')
    test_data = sio.loadmat('data\\test_32x32.mat')

    original_train = train_data['X']
    train_targets = train_data['y'].squeeze(1)
    print("y test", train_targets.shape)

    train_targets = np.array([x % 10 for x in train_targets])
    train_targets = torch.from_numpy(train_targets)

    # for target in train_targets:
    #     class_numbers[target] += 1

    X_train = preproccess_svhn(train_data['X'])
    print("x train shape", X_train.shape)

    # access to the dict

    X_test = preproccess_svhn(test_data['X'])
    print("x test shape", X_test.shape)

    targets = test_data['y'].squeeze(1)
    print("y test", targets.shape)

    targets = np.array([x % 10 for x in targets])
    targets = torch.from_numpy(targets)

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'supervised_queue_svhn_original'
    clusters_net_path = os.path.join(script_directory, filepath)

    encoder = DeepBinBrainCifar(3, EMBEDINGS).to('cuda')

    print(encoder)

    #print(list(encoder.brain[0].weight))
    #prune.random_unstructured(encoder.brain[0], name="weight", amount=0.6)
    #print(list(encoder.brain[0].weight))

    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    min_miss_percentage = 100
    max_loss_iter = 0

    test_best_loss = 1000

    print("X_train: ", X_train.shape, " X_test: ", X_test.shape, " targets: ", targets.shape)

    # test_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    # for idx, i in enumerate(targets):
    #     test_dict[i].append(idx)

    avg_loss = 0
    total_iters = 0

    for epoch in range(EPOCHS):
        print("Epoch: ", epoch)
        ids = np.random.choice(len(X_train), size=len(X_train), replace=False)

        runs = len(X_train) // BATCH_SIZE_DEFAULT

        rev_product = torch.ones([BATCH_SIZE_DEFAULT, EMBEDINGS]).cuda()
        old_ids = []
        first = True
        iteration = 0

        for j in range(runs):
            current_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
            iter_ids = ids[current_ids]

            encoder.train()

            train = True
            probs10, probs10_b, total_loss, rev_product, old_ids = forward_block(X_train, iter_ids, encoder, optimizer, train, rev_product, train_targets, old_ids)
            avg_loss += total_loss.item()
            iteration += 1
            total_iters += 1

            if iteration >= QUEUE:
                old_ids = old_ids[TR * BATCH_SIZE_DEFAULT:]
                rev_product = rev_product[TR * BATCH_SIZE_DEFAULT:, :]

        print("==================================================================================")

        print("batch mean ones: ",
              (np.where(probs10.data.cpu().numpy() > 0.5))[0].shape[0] / (probs10[0].shape[0] * BATCH_SIZE_DEFAULT))

        count_common_elements(probs10)
        print()

        print("train avg loss : ", avg_loss / runs)
        avg_loss = 0
        encoder.eval()

        test_loss = measure_acc_augments(X_test, encoder, rev_product, targets)

        if test_loss < test_best_loss:
            test_best_loss = test_loss
            max_loss_iter = total_iters
            min_miss_percentage = test_loss
            print("models saved iter: " + str(total_iters))
            torch.save(encoder, clusters_net_path + "_" + str(epoch//100) + '.model')

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