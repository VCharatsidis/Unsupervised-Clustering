"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch
from sklearn.datasets import fetch_openml
from MutualInfoMetric import MutualInfoMetric
from torch.autograd import Variable
import matplotlib.pyplot as plt
from detached_conv import DetachedConvNet
from sklearn.datasets import fetch_openml
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import copy
from predictor import Predictor
import statistics
from colon import Colon


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 5e-3
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 256
HALF_BATCH = BATCH_SIZE_DEFAULT // 2
EVAL_FREQ_DEFAULT = 400


FLAGS = None


def calc_distance(out, y):
    abs_difference = torch.abs(out - y)
    information_loss = torch.log(1 - abs_difference)

    mean = torch.mean(information_loss)

    # print(out)
    # print(y)
    # print(abs_difference)
    # print(mean)
    # print(torch.abs(mean))
    # input()

    return torch.abs(mean)


def flatten(out):
    return out.view(out.shape[0], -1)


def forward_block(X, ids, conv, colons, optimizers, train):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, 1)

    convolutions = conv.forward(x_tensor)

    flattened_convolutions = flatten(convolutions)

    size = flattened_convolutions.shape[1]

    total_loss = []

    for i in range(size):
        colon = colons[i]

        if train:
            colon.train()
        else:
            colon.eval()

        y = flattened_convolutions[:, i]
        x_reduced = torch.cat([flattened_convolutions[:, 0:i], flattened_convolutions[:, i + 1:]], 1)
        res = colon.forward(x_reduced.detach())
        distances = calc_distance(res, y)

        if train:
            optimizers[i].zero_grad()
            distances.backward(retain_graph=True)
            optimizers[i].step()

        total_loss.append(distances.item())

    return total_loss


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    conv = DetachedConvNet(1)

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'detached_net.model'
    detached_model = os.path.join(script_directory, filepath)
    torch.save(conv, detached_model)

    number_convolutions = 676
    colons = []
    optimizers = []
    colons_paths = []

    for i in range(number_convolutions):
        filepath = 'one_thousand_brains\\predictor_' + str(i) + '.model'
        predictor_model = os.path.join(script_directory, filepath)
        colons_paths.append(predictor_model)

        c = Colon(number_convolutions-1)
        colons.append(c)

        optimizer = torch.optim.SGD(c.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9)
        optimizers.append(optimizer)

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        loss_list = forward_block(X_train, ids, conv, colons, optimizers, train)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            print(loss_list)
            print(statistics.mean(loss_list))
            print(iteration)

            total_loss = 0

            test_size = 9984
            for i in range(BATCH_SIZE_DEFAULT, test_size, BATCH_SIZE_DEFAULT):
                ids = np.array(range(i - BATCH_SIZE_DEFAULT, i))
                loss_list = forward_block(X_test, ids, conv, colons, optimizers, False)

                total_loss += statistics.mean(loss_list)

            denom = test_size // BATCH_SIZE_DEFAULT
            total_loss = total_loss / denom

            if max_loss > total_loss:
                max_loss = total_loss
                print("models saved iter: " + str(iteration))
                for i in range(number_convolutions):
                    torch.save(colons[i], colons_paths[i])

            print("total loss " + str(total_loss))



def to_Tensor(X, batch_size=BATCH_SIZE_DEFAULT):
    X = np.reshape(X, (batch_size, 1, 28, 28))
    X = Variable(torch.FloatTensor(X))

    return X


def show_mnist(first_image):
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()



def main():
    """
    Main function
    """
    # Run the training operation
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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