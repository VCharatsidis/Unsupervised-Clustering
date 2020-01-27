from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch

from torch.autograd import Variable
import matplotlib.pyplot as plt
from stl10_input import read_all_images, read_labels
from encoderSTL import EncoderSTL
from stl_utils import rotate, scale, to_gray, vertical_flip, random_erease
from torchvision import transforms
import random


# Default constants
LEARNING_RATE_DEFAULT = 1e-7
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 50
EVAL_FREQ_DEFAULT = 200

FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def encode_4_patches(image, colons, train):
    image_1 = image[:, :, 10: 90, 10:90]
    image_1 = to_gray(image_1, 1)

    # image_1 = torch.transpose(image_1, 1, 3)
    # print(image_1.shape)
    # show_mnist(image_1[0], image_1[0].shape[1], image_1[0].shape[2])
    # image_1 = torch.transpose(image_1, 3, 1)
    #
    # image_1 = to_gray(image_1, 1)
    # print(image_1.shape)
    #
    # show_mnist(image_1[0][1], image_1[0].shape[0], image_1[0].shape[1])
    #
    # image_1 = image_1[:, 0, :, :]
    # show_mnist(image_1[0], image_1[0].shape[1], image_1[0].shape[1])
    #
    # image_1 = image_1[:, 2, :, :]
    # show_mnist(image_1[0], image_1[0].shape[1], image_1[0].shape[1])

    # augments = {0: to_gray(image_1, 1),
    #             1: rotate(image_1, 20),
    #             2: rotate(image_1, -20),
    #             3: scale(image_1, 60, 10),
    #             4: vertical_flip(image_1),
    #             5: scale(image_1, 40, 20, 20),
    #             6: random_erease(image_1),
    #             7: image_1}

    augments = {
                0: rotate(image_1, 20),
                1: rotate(image_1, -20),
                2: scale(image_1, 60, 10),
                3: vertical_flip(image_1),
                4: scale(image_1, 40, 20, 20),
                5: random_erease(image_1),
                6: image_1
    }

    ids = np.random.choice(len(augments), size=2, replace=False)
    image_1 = augments[ids[0]]
    image_2 = augments[ids[1]]

    if not train:
        augments_explain = {0: "gray",
                            1: "rotate 20",
                            2: "rotate -20",
                            3: "scale",
                            4: "vertical flip",
                            5: "more scale",
                            6: "random erase",
                            7: "original"}

        print(augments_explain[ids[0]])
        print(augments_explain[ids[1]])

    # print(image_1[0].shape)
    # image_1 = torch.transpose(image_1, 1, 3)
    # show_mnist(image_1[0], image_1[0].shape[1], image_1[0].shape[2])
    # image_1 = torch.transpose(image_1, 3, 1)
    #
    # image_2 = torch.transpose(image_2, 1, 3)
    # show_mnist(image_2[0], image_2[0].shape[1], image_2[0].shape[2])
    # image_2 = torch.transpose(image_2, 3, 1)

    image_1 = image_1.to('cuda')
    image_2 = image_2.to('cuda')

    pred_1 = colons[0](image_1)
    pred_2 = colons[1](image_2)

    return pred_1, pred_2


def forward_block(X, ids, colons, optimizers, train, to_tensor_size):
    x_train = X[ids, :]

    x_tensor = to_tensor(x_train, to_tensor_size)

    images = x_tensor/255.0

    pred_1, pred_2 = encode_4_patches(images, colons, train)

    product = pred_1 * pred_2
    product = product.mean(dim=0)
    log_product = torch.log(product)
    loss = - log_product.mean(dim=0)

    if train:
        optimizers[0].zero_grad()
        optimizers[1].zero_grad()

        loss.backward(retain_graph=True)

        optimizers[0].step()
        optimizers[1].step()

    return pred_1, pred_2, loss


def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():
    #
    fileName = "data\\stl10_binary\\train_X.bin"
    X_train = read_all_images(fileName)

    testFile = "data\\stl10_binary\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "data\\stl10_binary\\test_y.bin"
    targets = read_labels(test_y_File)


    # mnist = fetch_openml('mnist_784', version=1, cache=True)
    # targets = mnist.target[60000:]
    #
    # X_train = mnist.data[:60000]
    # X_test = mnist.data[60000:]

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    colons = []

    optimizers = []
    colons_paths = []

    filepath = 'encoders\\encoder_' + str(0) + '.model'
    predictor_model = os.path.join(script_directory, filepath)
    colons_paths.append(predictor_model)

    input = 18432
    #input = 1152

    # c = Ensemble()
    # c.cuda()

    c = EncoderSTL(3, input)
    c.cuda()
    colons.append(c)

    # c2 = EncoderSTL(3, input)
    # c2.cuda()
    # colons.append(c2)
    #
    # c3 = EncoderSTL(3, input)
    # c3.cuda()
    # colons.append(c3)
    #
    # c4 = EncoderSTL(3, input)
    # c4.cuda()
    # colons.append(c4)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    # optimizer2 = torch.optim.Adam(c2.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer2)
    #
    # optimizer3 = torch.optim.Adam(c3.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer3)
    #
    # optimizer4 = torch.optim.Adam(c4.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer4)

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:

            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)

            print()
            print("iteration: ", iteration)

            print(p1[0])
            print(p2[0])

            # dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
            dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 10: ""}
            dict = gather_data(p1, p2, targets, test_ids, dict)

            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)

            dict = gather_data(p1, p2, targets, test_ids, dict)

            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)

            dict = gather_data(p1, p2, targets, test_ids, dict)

            for i in dict.keys():
                print(i, " : ", dict[i])

            test_loss = mim.item()

            if max_loss > test_loss:
                max_loss = test_loss
                print("models saved iter: " + str(iteration))
                # for i in range(number_colons):
                #     torch.save(colons[i], colons_paths[i])

            print("test loss " + str(test_loss))
            print("")


def to_tensor(X, batch_size=BATCH_SIZE_DEFAULT):
    #X = np.reshape(X, (batch_size, 1, 96, 96))
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


def show_mnist(first_image, w, h):
    #pixels = first_image.reshape((w, h))
    plt.imshow(first_image)
    plt.show()


def gather_data(p1, p2, targets, test_ids, print_dict):

    for i in range(p1.shape[0]):
        if i == 10:
            print("")

        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + " , "

        label = targets[test_ids[i]]
        print_dict[label] += string

    return print_dict


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