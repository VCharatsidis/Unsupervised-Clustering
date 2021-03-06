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

from torch.autograd import Variable
import matplotlib.pyplot as plt

from base_conv import BaseConv
from sigmoid_layer import SigmoidLayer
from sklearn.datasets import fetch_openml

import statistics
from colon import Colon
from losses import IID_loss, mi

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 1
HALF_BATCH = BATCH_SIZE_DEFAULT // 2
EVAL_FREQ_DEFAULT = 1000

FLAGS = None


def calc_distance(out, y):
    abs_difference = torch.abs(out - y)
    eps = 1e-8
    information_loss = torch.log(1 - abs_difference + eps)

    mean = torch.mean(information_loss)

    return torch.abs(mean)


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def neighbours(convolutions):
    size = convolutions.shape[0]
    inputs = []

    for i in range(size):
        for j in range(size):
            conv_loss_tensor = torch.zeros(7, 7)

            for row in range(i-3, i+4):
                for col in range(j-3, j+4):
                    if row >= 0 and row < 28:
                        if col >= 0 and col < 28:
                            conv_loss_tensor[row - (i-3), col - (j-3)] = convolutions[row, col]

            flatten_input = torch.flatten(conv_loss_tensor)
            inputs.append(flatten_input)

    return inputs


def forward_block(X, ids, colons, optimizers, train, to_tensor_size):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, to_tensor_size)
    convolutions = x_tensor/255

    inputs = neighbours(convolutions[0, 0])

    flattened_convolutions = flatten(convolutions)
    size = flattened_convolutions.shape[1]

    colon_outputs = []

    for i in range(size):
        colon = colons[i]

        if train:
            colon.train()
        else:
            colon.eval()

        x_reduced = inputs[i]
        res = colon.forward(x_reduced.detach())
        colon_outputs.append(res)

    mutual_infos = []
    for idx1, i in enumerate(colon_outputs):
        for idx2, j in enumerate(colon_outputs):
            if idx1 >= idx2:
                continue

            loss = mi(i, j)
            mutual_infos.append(loss.item())
            if train:
                optimizers[idx1].zero_grad()
                loss.backward(retain_graph=True)
                optimizers[idx1].step()

                optimizers[idx2].zero_grad()
                loss.backward(retain_graph=True)
                optimizers[idx2].step()

    return colon_outputs, mutual_infos


def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    number_convolutions = 784

    script_directory = os.path.split(os.path.abspath(__file__))[0]


    colons = []
    optimizers = []
    colons_paths = []

    for i in range(number_convolutions):
        filepath = 'colons\\colon_' + str(i) + '.model'
        predictor_model = os.path.join(script_directory, filepath)
        colons_paths.append(predictor_model)

        c = Colon(49)
        colons.append(c)

        optimizer = torch.optim.SGD(c.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9)
        optimizers.append(optimizer)

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):
        if iteration % 50 == 0:
            print("iteration: ",iteration)

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        loss_list, mutual_infos = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            print("iteration: ", iteration)

            print(loss_list)
            print("mean: " + str(statistics.mean(loss_list)))

            total_loss = 0

            test_batch_size = 1048

            test_ids = np.random.choice(len(X_test), size=test_batch_size, replace=False)

            for c, i in enumerate(test_ids):
                if c % 100 == 0:
                    print("test iteration: "+str(c))
                loss_list, mutual_infos = forward_block(X_test, i, colons, optimizers, False, 1)

                total_loss += statistics.mean(mutual_infos)

            total_loss = total_loss / test_batch_size

            if max_loss > total_loss:
                max_loss = total_loss
                print("models saved iter: " + str(iteration))
                for i in range(number_convolutions):
                    torch.save(colons[i], colons_paths[i])

            print("total loss " + str(total_loss))
            print("")


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
