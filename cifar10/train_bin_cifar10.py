from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.utils.prune as prune

import sys

import argparse
import os

import cifar10_utils
from Bin_Cifar10_Net import BinCifar10Net
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

#EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 2.2e-4

MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 128
TR = 1

EMBEDINGS = 4096
SIZE = 32
SIZE_Y = 32
NETS = 1

EPOCHS = 300

CLASSES = 10
DESCRIPTION = " Image size: " + str(SIZE) + " , Classes: " + str(CLASSES)
QUEUE = 380

EVAL_FREQ_DEFAULT = 250
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None

square = torch.ones(BATCH_SIZE_DEFAULT, BATCH_SIZE_DEFAULT)
ZERO_DIAG = square.fill_diagonal_(0)
first_part = torch.cat([ZERO_DIAG, ZERO_DIAG, ZERO_DIAG, ZERO_DIAG], dim=1)
adj_matrix = torch.cat([first_part, first_part, first_part, first_part], dim=0)
adj_matrix = adj_matrix.cuda()

# big_diag = torch.ones(TR * BATCH_SIZE_DEFAULT, TR * BATCH_SIZE_DEFAULT)
# ZERO_BIG_DIAG = big_diag.fill_diagonal_(0)
# ZERO_BIG_DIAG = ZERO_BIG_DIAG.cuda()

#ELEMENTS_EXCEPT_DIAG = 2 * BATCH_SIZE_DEFAULT * (BATCH_SIZE_DEFAULT - 1)

first = True


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


def save_images(images, transformation):
    print(transformations_dict[transformation])
    numpy_cluster = images.cpu().detach()
    save_cluster(numpy_cluster, transformations_dict[transformation], 0)


def new_agreement(product, denominator, rev_prod):

    attraction = (torch.mm(product, product.transpose(0, 1)) + EPS) * (1-adj_matrix)
    repel = (torch.mm(product, rev_prod.transpose(0, 1)) + EPS) * adj_matrix

    attraction = attraction / denominator
    repel = repel / denominator

    total_matrix = - torch.log(repel + attraction)

    # attraction_bonus = 0.5 * (BATCH_SIZE_DEFAULT - 1) * (1-adj_matrix) * total_matrix
    #
    # total_matrix = total_matrix + attraction_bonus.fill_diagonal_(0)

    mean_total = total_matrix.mean()

    return mean_total


def queue_agreement(product, denominator, rev_prod):
    transposed = rev_prod.transpose(0, 1)

    nondiag = torch.mm(product, transposed) + EPS
    nondiag = nondiag / denominator

    log_nondiag = - torch.log(nondiag)
    negative = log_nondiag.mean()

    return negative


def make_transformations(image, aug_ids, iter):

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

    # save_images(image, 0)
    # save_images(image_1_a, aug_ids[0])
    # save_images(image_1_b, aug_ids[1])
    # save_images(image_1_c, aug_ids[2])
    #
    # save_images(image_2_a, aug_ids[3])
    # save_images(image_2_b, aug_ids[4])
    # save_images(image_2_c, aug_ids[5])
    # save_images(image_2_d, aug_ids[6])
    #
    # save_images(image_3_b, aug_ids[7])
    # save_images(image_3_c, aug_ids[8])
    # save_images(image_3_d, aug_ids[9])
    #
    # save_images(image_4_a, aug_ids[10])
    # save_images(image_4_c, aug_ids[11])
    #
    # save_images(image_5_a, aug_ids[12])
    # save_images(image_5_b, aug_ids[13])
    # save_images(image_5_c, aug_ids[14])
    #
    # save_images(image_6_a, aug_ids[15])
    #
    # save_images(image_4_d, aug_ids[16])
    # save_images(image_8_d, aug_ids[17])
    # save_images(image_8_c, 18)

    image_1 = torch.cat([image_1_a, image_2_a, image_3_a, image_4_a, image_5_a, image_6_a, image_7_a, image_8_a], dim=0)
    image_2 = torch.cat([image_1_b, image_2_b, image_3_b, image_4_b, image_5_b, image_6_b, image_7_b, image_8_b], dim=0)
    image_3 = torch.cat([image_1_c, image_2_c, image_3_c, image_4_c, image_5_c, image_6_c, image_7_c, image_8_c], dim=0)
    image_4 = torch.cat([image_1_d, image_2_d, image_3_d, image_4_d, image_5_d, image_6_d, image_7_d, image_8_d], dim=0)

    return image_1, image_2, image_3, image_4


def forward_block(X, ids, encoder, optimizer, train, rev_product):
    global first
    number_transforms = 19
    aug_ids = np.random.choice(number_transforms, size=number_transforms, replace=False)

    image = X[ids, :]
    image_1, image_2, image_3, image_4 = make_transformations(image, aug_ids, 0)

    _, logit_a, a = encoder(image_1.to('cuda'))
    _, logit_b, b = encoder(image_2.to('cuda'))
    _, logit_c, c = encoder(image_3.to('cuda'))
    _, logit_d, d = encoder(image_4.to('cuda'))

    all_predictions = torch.cat([a, b, c, d], dim=0)

    to_store = 1 - a
    current_reverse = 1 - all_predictions
    denominator = torch.cat([a.sum(dim=1), b.sum(dim=1), c.sum(dim=1), d.sum(dim=1)], dim=0)
    denominator = denominator.unsqueeze(dim=1) + 1

    new_loss = new_agreement(all_predictions, denominator, current_reverse)

    if first or not train:
        total_loss = new_loss
        rev_product = to_store.detach()
        first = False

    else:
        old_loss = queue_agreement(all_predictions, denominator, rev_product)
        rev_product = torch.cat([rev_product, to_store.detach()])
        total_loss = new_loss + old_loss

    if train:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return a, b, total_loss, rev_product


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def measure_acc_augments(x_test, encoder, rev_product):
    size = BATCH_SIZE_DEFAULT
    runs = len(x_test) // size
    sum_loss = 0

    print(rev_product.shape)

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)

        with torch.no_grad():
            test_preds_1, test_preds_2, test_total_loss, rev_product = forward_block(x_test, test_ids, encoder, [], False, rev_product)

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
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = cifar10_utils.load_cifar10(cifar10_utils.CIFAR10_FOLDER)
    X_train, y_train, X_test, targets = cifar10_utils.preprocess_cifar10_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw)

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)

    X_train /= 255
    X_test /= 255

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'binary_contrast_4_32'
    clusters_net_path = os.path.join(script_directory, filepath)

    #encoder = torch.load(filepath + "_1.model")

    encoder = BinCifar10Net(3, EMBEDINGS).to('cuda')

    #torch.save(encoder, os.path.join(script_directory, "cifar100_models\\a_bcnet_random_net.model"))
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
        first = True
        iteration = 0

        for j in range(runs):
            current_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
            encoder.train()
            iter_ids = ids[current_ids]

            train = True
            probs10, probs10_b, total_loss, rev_product = forward_block(X_train, iter_ids, encoder, optimizer, train, rev_product)
            avg_loss += total_loss.item()
            iteration += 1
            total_iters += 1

            if iteration >= QUEUE:
                rev_product = rev_product[TR * BATCH_SIZE_DEFAULT:, :]

        print("==================================================================================")
        # print("example prediction: ", probs10[0])
        # print("example prediction: ", probs10[1])
        print("batch mean ones: ",
              (np.where(probs10.data.cpu().numpy() > 0.5))[0].shape[0] / (probs10[0].shape[0] * BATCH_SIZE_DEFAULT))

        count_common_elements(probs10)
        print()

        print("train avg loss : ", avg_loss / runs)
        avg_loss = 0
        encoder.eval()

        test_loss = measure_acc_augments(X_test, encoder, rev_product)

        if test_loss < test_best_loss:
            test_best_loss = test_loss
            max_loss_iter = total_iters
            min_miss_percentage = test_loss
            print("models saved iter: " + str(total_iters))

            torch.save(encoder, clusters_net_path+"_"+str(epoch // 100)+".model")

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


def to_tensor(X):
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


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