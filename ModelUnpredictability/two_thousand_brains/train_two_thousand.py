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
from conv import ConvNet
from sklearn.datasets import fetch_openml
import statistics
from colon import Colon


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 256
HALF_BATCH = BATCH_SIZE_DEFAULT // 2
EVAL_FREQ_DEFAULT = 200


FLAGS = None


def calc_distance(out, y):
    # print("out")
    # print(out)
    # print("y")
    # print(y)
    abs_difference = torch.abs(out - y)
    # print("abs_difference")
    # print(abs_difference)

    z = 1 - abs_difference
    information_loss = torch.log(z)
    # print("information_loss")
    # print(information_loss)

    mean_info = torch.mean(information_loss, 1)
    # print("mean_info")
    # print(mean_info)

    mean = torch.mean(information_loss)
    # print("mean")
    # print(mean)


    return torch.abs(mean), mean_info


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def forward_block(X, ids, conv, colons, optimizers, conv_second, colons_second, optimizers_second, train):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train,1)

    convolutions = conv.forward(x_tensor)

    flattened_convolutions = flatten(convolutions)

    size = flattened_convolutions.shape[1]

    total_loss = []

    concat = torch.tensor([])
    for i in range(size):
        colon = colons[i]

        if train:
            colon.train()
        else:
            colon.eval()

        y = flattened_convolutions[:, i]
        x_reduced = torch.cat([flattened_convolutions[:, 0:i], flattened_convolutions[:, i + 1:]], 1)
        res = colon.forward(x_reduced.detach())
        distances, information_loss = calc_distance(res, y)

        #print(information_loss)
        info_unsq = information_loss.unsqueeze_(1)
        #print(information_loss)
        info_unsq = info_unsq.transpose(0, 1)

        concat = torch.cat([concat, info_unsq])
        #print(concat)

        if train:
            optimizers[i].zero_grad()
            distances.backward(retain_graph=True)
            optimizers[i].step()

        total_loss.append(distances.item())

    concat = concat.transpose(1, 0)

    x_tensor_second = to_Tensor(concat.detach().numpy(), 1, 26, 26)
    convolutions_second = conv_second.forward(x_tensor_second)
    flattened_convolutions_second = flatten(convolutions_second)
    size_second = flattened_convolutions_second.shape[1]

    total_loss_second = []

    for i in range(size_second):
        colon_second = colons_second[i]

        if train:
            colon_second.train()
        else:
            colon_second.eval()

        y_second = flattened_convolutions_second[:, i]
        x_reduced_second = torch.cat([flattened_convolutions_second[:, 0:i], flattened_convolutions_second[:, i + 1:]], 1)

        res_second = colon_second.forward(x_reduced_second.detach())
        distances_second, _ = calc_distance(res_second, y_second)

        if train:
            optimizers_second[i].zero_grad()
            distances_second.backward(retain_graph=True)
            optimizers_second[i].step()

        total_loss_second.append(distances_second.item())

    return total_loss_second


def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    number_convolutions = 676

    stride = 1
    if number_convolutions == 169:
        stride = 2

    filters = 1
    conv = ConvNet(1, filters, stride)

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'conv_net.model'
    conv_model = os.path.join(script_directory, filepath)
    torch.save(conv, conv_model)

    conv_second = ConvNet(1, filters, 1)

    filepath = 'conv_net_second.model'
    conv_second_model = os.path.join(script_directory, filepath)
    torch.save(conv_second, conv_second_model)

    colons = []
    optimizers = []
    colons_paths = []

    for i in range(number_convolutions):
        filepath = 'predictors\\predictor_' + str(i) + '.model'
        predictor_model = os.path.join(script_directory, filepath)
        colons_paths.append(predictor_model)

        c = Colon(filters * (number_convolutions-1))
        colons.append(c)

        optimizer = torch.optim.SGD(c.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9)
        optimizers.append(optimizer)

    colons_second = []
    optimizers_second = []
    colons_paths_second = []

    number_convolutions_second = 576
    for i in range(number_convolutions_second):
        filepath_second = 'second_predictors\\second_predictor_' + str(i) + '.model'
        predictor_model_second = os.path.join(script_directory, filepath_second)
        colons_paths_second.append(predictor_model_second)

        c_second = Colon(filters * (number_convolutions_second-1))
        colons_second.append(c_second)

        optimizer_second = torch.optim.SGD(c_second.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9)
        optimizers_second.append(optimizer_second)

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        loss_list = forward_block(X_train, ids, conv, colons, optimizers, conv_second, colons_second, optimizers_second, train)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            print_params(conv)
            print(loss_list)
            print(statistics.mean(loss_list))
            print(iteration)

            total_loss = 0

            test_size = 9984
            for i in range(BATCH_SIZE_DEFAULT, test_size, BATCH_SIZE_DEFAULT):
                ids = np.array(range(i - BATCH_SIZE_DEFAULT, i))
                loss_list = forward_block(X_test, ids, conv, colons, optimizers, conv_second, colons_second, optimizers_second, False)

                total_loss += statistics.mean(loss_list)

            denom = test_size // BATCH_SIZE_DEFAULT
            total_loss = total_loss / denom

            if max_loss > total_loss:
                max_loss = total_loss
                print("models saved iter: " + str(iteration))
                torch.save(conv, conv_model)
                torch.save(conv_second, conv_second_model)
                for i in range(number_convolutions):
                    torch.save(colons[i], colons_paths[i])

                for i in range(number_convolutions_second):
                    torch.save(colons_second[i], colons_paths_second[i])

            print("total loss " + str(total_loss))
            print("")


def to_Tensor(X, batch_size=BATCH_SIZE_DEFAULT, width=28, height=28):
    X = np.reshape(X, (batch_size, 1, width, height))
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