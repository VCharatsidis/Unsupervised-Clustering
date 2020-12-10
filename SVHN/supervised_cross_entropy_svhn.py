from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.utils.prune as prune

import sys
import scipy.io as sio
import argparse
import os

from SupervisedSVHNNET import SupervisedSVHNNET

from image_utils import *
import random

from torchvision.utils import make_grid
import matplotlib
import pickle

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

EPS = sys.float_info.epsilon

#EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 4e-4

MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 256

SIZE = 32
SIZE_Y = 32
NETS = 1

EPOCHS = 400

CLASSES = 10
DESCRIPTION = " Image size: " + str(SIZE) + " , Classes: " + str(CLASSES)

EVAL_FREQ_DEFAULT = 250
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None

loss = nn.CrossEntropyLoss()
# square = torch.ones(BATCH_SIZE_DEFAULT, BATCH_SIZE_DEFAULT)
# ZERO_DIAG = square.fill_diagonal_(0)

ELEMENTS_EXCEPT_DIAG = BATCH_SIZE_DEFAULT * (BATCH_SIZE_DEFAULT - 1)

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
                       8: "random crop",
                       9: "random crop",
                       10: "random crop",
                       11: "random crop",
                       12: "image 1",
                       13: "image 2"}


def save_images(images, transformation):
    print(transformations_dict[transformation])
    numpy_cluster = images.cpu().detach()
    save_cluster(numpy_cluster, transformations_dict[transformation], 0)


# def new_agreement(product, denominator, rev_prod):
#     transposed = rev_prod.transpose(0, 1)
#
#     nondiag = torch.mm(product, transposed)
#     nondiag = nondiag / denominator.unsqueeze(dim=1)
#
#     log_nondiag = - torch.log(nondiag)
#
#     negative = log_nondiag.mean()
#
#     return negative
#
#
# def queue_agreement(product, denominator, rev_prod):
#     transposed = rev_prod.transpose(0, 1)
#
#     nondiag = torch.mm(product, transposed)
#     nondiag = nondiag / denominator.unsqueeze(dim=1)
#
#     log_nondiag = - torch.log(nondiag)
#     negative = log_nondiag.mean()
#
#     return negative


# def forward_block(X, y_train, ids, encoder, optimizer, train, rev_product):
#     global first
#
#     image = X[ids, :]
#
#     _, _, a = encoder(image.to('cuda'))
#
#     y_train = Variable(torch.LongTensor(y_train[ids])).cuda()
#
#     cross_entropy_loss = loss(a, y_train)
#
#     if train:
#         optimizer.zero_grad()
#         cross_entropy_loss.backward()
#         optimizer.step()
#
#     return a, a, cross_entropy_loss, rev_product


def forward_block(X, y_train, ids, encoder, optimizer, train, rev_product):
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

    #image_1 = torch.cat([image_1, image_3, image_5, image_7, image_9, image_11, image_13, image_15], dim=0)
    #image_2 = torch.cat([image_2, image_4, image_6, image_8, image_10, image_12, image_14, image_16], dim=0)

    # save_images(image_1, 20)
    # save_images(image_2, 21)

    _, _, a = encoder(image.to('cuda'))
    #_, _, b = encoder(image_2.to('cuda'))

    #all_predictions = torch.cat([a, b], dim=0)
    all_predictions = a

    y_train = Variable(torch.LongTensor(y_train[ids])).cuda()
    #current_targets = torch.cat([y_train, y_train], dim=0)
    current_targets = y_train

    cross_entropy_loss = loss(all_predictions, current_targets)

    if train:
        optimizer.zero_grad()
        cross_entropy_loss.backward()
        optimizer.step()

    return a, b, cross_entropy_loss, rev_product


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
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
            test_preds_1, test_preds_2, test_total_loss, rev_product = forward_block(x_test, targets, test_ids, encoder, [], False, rev_product)

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
    # unsupervised_data = sio.loadmat('data\\extra_32x32.mat')
    train_data = sio.loadmat('data\\train_32x32.mat')
    test_data = sio.loadmat('data\\test_32x32.mat')

    train_targets = train_data['y'].squeeze(1)
    print("y test", train_targets.shape)

    y_train = np.array([x % 10 for x in train_targets])
    #y_train = torch.from_numpy(train_targets)

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
    #targets = torch.from_numpy(targets)

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'supervised_cross_entropy_svhn_original'
    clusters_net_path = os.path.join(script_directory, filepath)

    encoder = SupervisedSVHNNET(3, 10).to('cuda')

    print(encoder)

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

        rev_product = torch.ones([BATCH_SIZE_DEFAULT, 1])
        first = True
        iteration = 0

        for j in range(runs):
            current_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
            encoder.train()
            iter_ids = ids[current_ids]

            train = True
            probs10, probs10_b, total_loss, rev_product = forward_block(X_train, y_train, iter_ids, encoder, optimizer, train, rev_product)
            avg_loss += total_loss.item()
            iteration += 1
            total_iters += 1

            # if iteration >= 50:
            #     rev_product = rev_product[BATCH_SIZE_DEFAULT:, :]

        print("==================================================================================")

        print("train avg loss : ", avg_loss / runs)
        avg_loss = 0
        encoder.eval()

        test_loss = measure_acc_augments(X_test, encoder, rev_product, targets)

        if test_loss < test_best_loss:
            test_best_loss = test_loss
            max_loss_iter = total_iters
            min_miss_percentage = test_loss
            print("models saved iter: " + str(total_iters))
            torch.save(encoder, clusters_net_path + "_" + str(epoch//100) + ".model")

        print("EPOCH: ", epoch,
              "Total ITERATION: ", total_iters,
              " epoch iter: ", iteration,
              ",  batch size: ", BATCH_SIZE_DEFAULT,
              ",  lr: ", LEARNING_RATE_DEFAULT,
              ",  best loss iter: ", max_loss_iter,
              "-", min_miss_percentage
              , DESCRIPTION)



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