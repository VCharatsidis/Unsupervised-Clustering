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

EMBEDINGS = 256
SIZE = 32
SIZE_Y = 32
NETS = 1

EPOCHS = 1000

CLASSES = 100
DESCRIPTION = " Image size: " + str(SIZE) + " , Classes: " + str(CLASSES)

EVAL_FREQ_DEFAULT = 200
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None

square = torch.ones(BATCH_SIZE_DEFAULT, BATCH_SIZE_DEFAULT)
ZERO_DIAG = square.fill_diagonal_(0)
first_part = torch.cat([ZERO_DIAG, ZERO_DIAG], dim=1)
adj_matrix = torch.cat([first_part, first_part], dim=0)
adj_matrix = adj_matrix.cuda()

big_diag = torch.ones(2 * BATCH_SIZE_DEFAULT, 2 * BATCH_SIZE_DEFAULT)
ZERO_BIG_DIAG = big_diag.fill_diagonal_(0).cuda()


ELEMENTS_EXCEPT_DIAG = 2 * BATCH_SIZE_DEFAULT * (BATCH_SIZE_DEFAULT - 1)

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
    transposed = rev_prod.transpose(0, 1)

    attraction = torch.mm(product, product.transpose(0, 1))
    repel = torch.mm(product, transposed)

    denominator = denominator.unsqueeze(dim=1)

    attraction = attraction / denominator
    repel = repel / denominator

    total_matrix = repel * class_adj_matrix + attraction * (1 - class_adj_matrix)
    total_matrix[(total_matrix < EPS).data] = EPS

    log_total = - torch.log(total_matrix)

    diagonal_elements_bonus = ZERO_BIG_DIAG * 5 * (1 - class_adj_matrix) * log_total

    total = log_total + diagonal_elements_bonus

    total = total.mean()

    # zero_out_same_class_elements = total * class_adj_matrix.detach().cuda()
    #
    # sum_non_zero_elements = class_adj_matrix.detach().sum()
    #
    # total_cleaned = zero_out_same_class_elements.sum() / sum_non_zero_elements.cuda()

    return total


def queue_agreement(product, denominator, rev_prod):
    transposed = rev_prod.transpose(0, 1)

    nondiag = torch.mm(product, transposed)
    nondiag = nondiag / denominator.unsqueeze(dim=1)

    log_nondiag = - torch.log(nondiag)
    negative = log_nondiag.mean()

    return negative


def forward_block(X, ids, encoder, optimizer, train, rev_product, moving_mean):
    global first
    number_transforms = 19
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

    _, c_a, a = encoder(image_1.to('cuda'))
    _, c_b, b = encoder(image_2.to('cuda'))

    all_predictions = torch.cat([a, b], dim=0)

    current_reverse = 1 - all_predictions
    denominator = torch.cat([a.sum(dim=1), b.sum(dim=1)], dim=0)

    labels, max_elems = assign_labels(a, b)

    #round_max = torch.round(max_elems).mean()
    classification_loss, current_mean = class_mean_probs_loss(c_a, c_b, moving_mean)

    # max_mean, _ = moving_mean.max(dim=0)
    # current_max, _ = current_mean.max(dim=0)

    # if max_mean > 1.5 * moving_mean.mean() or max_mean < 0.5 or current_max > 10 * current_mean.mean():
    #     class_adj_m = torch.ones(2*BATCH_SIZE_DEFAULT, 2*BATCH_SIZE_DEFAULT)
    # else:
    #     print("mask used")
    #     class_adj_m = class_adj_matrix(labels)

    class_adj_m = class_adj_matrix(labels)
    new_loss = new_agreement(all_predictions, denominator, current_reverse, class_adj_m)

    if first or not train:
        # print("first: ", first)
        total_loss = new_loss + classification_loss
        # rev_product = current_reverse.detach()
        first = True

    # The else is obsolete
    else:
        print("queue_agreement")
        old_loss = queue_agreement(all_predictions, denominator, rev_product)
        rev_product = torch.cat([rev_product, current_reverse.detach()])
        total_loss = new_loss + old_loss

    if train:
        moving_mean = 0.95 * moving_mean.cuda() + 0.05 * current_mean.detach()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return a, b, total_loss, rev_product, classification_loss, moving_mean


def assign_labels(a, b):
    predictions = (a + b) / 2

    max_elems, indexes = predictions.max(dim=1)

    # print(max_elems.shape)
    # print(indexes.shape)
    #

    # #print(predictions[0])
    # print(indexes[0])

    return indexes, max_elems


def class_adj_matrix(labels):
    square = torch.ones(BATCH_SIZE_DEFAULT, BATCH_SIZE_DEFAULT)

    labels_vertical = labels.unsqueeze(dim=1)
    square[(labels == labels_vertical).data] = 0

    first_part = torch.cat([square, square], dim=1)
    class_adj_matrix = torch.cat([first_part, first_part], dim=0)

    return class_adj_matrix.cuda()


def class_mean_probs_loss(a, b, moving_mean):
    product = a * b
    current_mean = product.mean(dim=0)

    batch_sam = (a.mean(dim=0) + b.mean(dim=0)) / 2
    result = current_mean / batch_sam

    interpolation = moving_mean.detach().cuda() * 0.8 + result * 0.2

    log = - torch.log(interpolation + EPS)
    scalar = log.mean()

    return scalar, result


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def measure_acc_augments(x_test, encoder, rev_product, moving_mean):
    size = BATCH_SIZE_DEFAULT
    runs = len(x_test) // size
    sum_loss = 0
    sum_class_loss = 0
    print(rev_product.shape)

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)

        with torch.no_grad():
            test_preds_1, test_preds_2, test_total_loss, rev_product, class_loss, moving_mean = forward_block(x_test, test_ids, encoder, [], False, rev_product, moving_mean)

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


    moving_mean = torch.ones(CLASSES) / (100*CLASSES)

    avg_loss = 0
    total_iters = 0
    avg_class_loss = 0

    for epoch in range(EPOCHS):
        print("Epoch: ", epoch)
        ids = np.random.choice(len(X_train), size=len(X_train), replace=False)

        runs = len(X_train) // BATCH_SIZE_DEFAULT

        rev_product = torch.ones([BATCH_SIZE_DEFAULT, EMBEDINGS]).cuda()
        first = True
        iteration = 0

        for j in range(runs):
            current_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
            encoder.train()
            iter_ids = ids[current_ids]

            train = True
            probs10, probs10_b, total_loss, rev_product, class_loss, moving_mean = forward_block(X_train, iter_ids, encoder, optimizer, train, rev_product, moving_mean)
            avg_loss += total_loss.item()
            avg_class_loss += class_loss.item()
            iteration += 1
            total_iters += 1

            if iteration >= 40:
                rev_product = rev_product[4 * BATCH_SIZE_DEFAULT:, :]

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

        test_loss = measure_acc_augments(X_test, encoder, rev_product, moving_mean)

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