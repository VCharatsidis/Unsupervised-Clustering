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
from decoder import Decoder

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
    abs_difference = torch.min(torch.ones(abs_difference.shape), abs_difference)
    abs_difference = torch.max(torch.zeros(abs_difference.shape), abs_difference)

    eps = 1e-8
    information_loss = torch.log(1 - abs_difference + eps)

    mean = torch.sum(information_loss)

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


def forward_decoder(X, ids, base_conv, decoder, optimizer, train):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, 1)
    image = x_tensor / 255
    # show_mnist(image)

    convolved_image = base_conv.forward(image)
    flatten_convolved_image = flatten(convolved_image)

    flattened_image = flatten(image)

    colon_outputs = []

    if train:
        decoder.train()
    else:
        decoder.eval()

    res = decoder.forward(flatten_convolved_image)
    colon_outputs.append(res)
    distance = calc_distance(res, flattened_image)

    if train:
        optimizer.zero_grad()
        distance.backward(retain_graph=True)
        optimizer.step()

    return distance, res, flatten_convolved_image


def forward_block(X, ids, conv, base_conv, colons, optimizers, train, to_tensor_size):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, to_tensor_size)
    image = x_tensor/255
    # show_mnist(image)
    simple_convolutions = conv.forward(image)

    flatten_simple_conv = flatten(simple_convolutions)
    # show_mnist(flatten_simple_conv.detach().numpy())

    convolved_image = base_conv.forward(image)
    flatten_convolved_image = flatten(convolved_image)

    #inputs = neighbours(image[0, 0])

    flattened_image = flatten(image)
    size = flatten_simple_conv.shape[1]

    # base_convolutions = base_conv.forward(x_tensor)
    # flatten_base_conv = flatten(base_convolutions)

    total_loss = []
    conv_loss_tensor = torch.zeros(size)
    colon_outputs = []

    for i in range(size):
        colon = colons[i]

        if train:
            colon.train()
        else:
            colon.eval()

        y = flattened_image[:, i]

        #torch_cat = torch.cat([flattened_convolutions[:, 0:i], flattened_convolutions[:, i + 1:]], 1)

        #x_reduced = inputs[i] #flatten_base_conv # torch.cat([flattened_convolutions[:, 0:i], flattened_convolutions[:, i + 1:]], 1)

        x_reduced = flatten_convolved_image
        res = colon.forward(x_reduced)
        colon_outputs.append(res)
        distances = calc_distance(res, y)

        if train:
            optimizers[i].zero_grad()
            distances.backward(retain_graph=True)
            optimizers[i].step()

        conv_loss_tensor[i] = distances
        total_loss.append(distances.item())

    conv_loss_tensor = torch.min(torch.ones(conv_loss_tensor.shape), conv_loss_tensor)
    conv_loss_tensor = torch.max(torch.zeros(conv_loss_tensor.shape), conv_loss_tensor)

    eps = 1e-8
    log_total_loss = torch.log(1 - conv_loss_tensor + eps)
    conv_loss = torch.sum(log_total_loss)
    abs_conv_loss = torch.abs(conv_loss)

    if train:
        optimizers[-1].zero_grad()
        abs_conv_loss.backward()
        optimizers[-1].step()

    return total_loss, abs_conv_loss.item(), colon_outputs, flatten_convolved_image


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

    stride = 1
    if number_convolutions == 169:
        stride = 2

    filters = 1
    conv = SigmoidLayer()
    base_conv = BaseConv(1)
    decoder = Decoder()

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'detached_net.model'
    detached_model = os.path.join(script_directory, filepath)
    torch.save(conv, detached_model)

    filepath = 'base_conv.model'
    base_conv_model = os.path.join(script_directory, filepath)
    torch.save(base_conv, base_conv_model)

    filepath = 'decoder.model'
    decoder_model = os.path.join(script_directory, filepath)
    torch.save(decoder, decoder_model)

    colons = []
    optimizers = []
    colons_paths = []

    for i in range(number_convolutions):
        filepath = 'colons\\colon_' + str(i) + '.model'
        predictor_model = os.path.join(script_directory, filepath)
        colons_paths.append(predictor_model)

        c = Colon(10)
        colons.append(c)

        params = list(c.parameters())
        optimizer = torch.optim.SGD(params, lr=LEARNING_RATE_DEFAULT, momentum=0.9)
        optimizers.append(optimizer)

    params = list(base_conv.parameters()) + list(decoder.parameters())
    base_conv_optimizer = torch.optim.Adam(params, lr=1e-3)
    optimizers.append(base_conv_optimizer)
    max_loss = 1999

    # all_train_inputs = []
    # for x in X_train:
    #     x_tensor = to_Tensor(x, BATCH_SIZE_DEFAULT)
    #     convolutions = x_tensor / 255
    #     input = neighbours(convolutions)
    #     all_train_inputs.append(input)
    #
    # all_test_inputs = []
    # for x in X_test:
    #     x_tensor = to_Tensor(x, BATCH_SIZE_DEFAULT)
    #     convolutions = x_tensor / 255
    #     input = neighbours(convolutions)
    #     all_test_inputs.append(input)

    for iteration in range(MAX_STEPS_DEFAULT):
        if iteration % 100 == 0:
            # print_params(base_conv)
            print("iteration: ", iteration)

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        loss_list, base_conv_loss, _, _ = forward_block(X_train, ids, conv, base_conv, colons, optimizers, train, BATCH_SIZE_DEFAULT)

        #loss_list, res, _ = forward_decoder(X_train, ids, base_conv, decoder, base_conv_optimizer, train)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            print("iteration: ", iteration)
            # print("convolution loss: " + str(base_conv_loss))
            # numpy_loss = np.array(loss_list)
            # show_mnist(numpy_loss)
            # print_params(conv)
            print(loss_list)
            print("mean: " + str(statistics.mean(loss_list)))

            total_loss = 0

            test_batch_size = 500

            test_ids = np.random.choice(len(X_test), size=test_batch_size, replace=False)

            for c, i in enumerate(test_ids):
                if c % 100 == 0:
                    print("test iteration: "+str(c))
                loss_list, _, _, _ = forward_block(X_test, i, conv, base_conv, colons, optimizers, False, 1)
                #loss_list, res, _ = forward_decoder(X_test, i, base_conv, decoder, base_conv_optimizer, False)

                total_loss += statistics.mean(loss_list)
                #total_loss += loss_list

            total_loss = total_loss / test_batch_size

            if max_loss > total_loss:
                max_loss = total_loss
                print("models saved iter: " + str(iteration))
                torch.save(conv, detached_model)
                torch.save(base_conv, base_conv_model)
                torch.save(decoder, decoder_model)
                # for i in range(number_convolutions):
                #     torch.save(colons[i], colons_paths[i])

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
