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
from encoder import Encoder
from decoder import Decoder
from guesser import Guesser

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


def neighbours(convolutions, perimeter):
    size = convolutions.shape[0]
    inputs = []

    for i in range(size):
        for j in range(size):
            total_input = 2 * perimeter + 1
            conv_loss_tensor = torch.zeros(total_input, total_input)

            for row in range(i-perimeter, i+perimeter+1):
                for col in range(j-perimeter, j+perimeter+1):
                    if row >= 0 and row < size:
                        if col >= 0 and col < size:
                            conv_loss_tensor[row - (i-perimeter), col - (j-perimeter)] = convolutions[row, col]

            flatten_input = torch.flatten(conv_loss_tensor)
            inputs.append(flatten_input)

    return inputs


def forward_decoder(X, ids, encoder, guesser, decoder, optimizer, train):
    if train:
        decoder.train()
        guesser.train()
        encoder.train()
    else:
        decoder.eval()
        guesser.eval()
        encoder.eval()

    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, 1)
    image = x_tensor / 255

    encoded_image = encoder.forward(image)
    flatten_convolved_image = flatten(encoded_image)

    # Guess the convolved here

    inputs = neighbours(encoded_image, 1)
    losses = torch.zeros(len(inputs))

    for i in range(len(inputs)):
        guess = guesser(inputs[i])
        losses[i] = calc_distance(guess, flatten_convolved_image[i])

    sum_guess_loss = torch.sum(losses)
    sum_guess_loss = torch.abs(sum_guess_loss)

    flattened_image = flatten(image)

    res = decoder.forward(flatten_convolved_image)
    distance = calc_distance(res, flattened_image)

    total_loss = sum_guess_loss + distance

    if train:
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

    return total_loss, distance, res, flatten_convolved_image


def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    guesser = Guesser()
    encoder = Encoder(1)
    decoder = Decoder()

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'guesser.model'
    guesser_model = os.path.join(script_directory, filepath)
    torch.save(guesser, guesser_model)

    filepath = 'encoder.model'
    encoder_model = os.path.join(script_directory, filepath)
    torch.save(encoder, encoder_model)

    filepath = 'decoder.model'
    decoder_model = os.path.join(script_directory, filepath)
    torch.save(decoder, decoder_model)

    optimizers = []

    params = list(encoder.parameters()) + list(decoder.parameters())
    base_conv_optimizer = torch.optim.Adam(params, lr=1e-3)
    optimizers.append(base_conv_optimizer)
    max_loss = 10000000

    for iteration in range(MAX_STEPS_DEFAULT):
        if iteration % 100 == 0:
            # print_params(base_conv)
            print("iteration: ", iteration)

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        loss_list, distance, res, flatten_convolved_image = forward_decoder(X_train, ids, encoder, guesser, decoder, optimizers, train)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            print("iteration: ", iteration)
            #print("convolution loss: " + str(base_conv_loss))
            # numpy_loss = np.array(loss_list)
            # show_mnist(numpy_loss)
            # print_params(conv)
            print(loss_list)
            #print("mean: " + str(statistics.mean(loss_list)))

            total_loss = 0

            test_batch_size = 500

            test_ids = np.random.choice(len(X_test), size=test_batch_size, replace=False)

            for c, i in enumerate(test_ids):
                if c % 100 == 0:
                    print("test iteration: "+str(c))
                loss_list, distance, res, flatten_convolved_image = forward_decoder(X_test, ids, encoder, guesser, decoder, optimizers, False)

                #total_loss += statistics.mean(loss_list)
                total_loss += loss_list

            total_loss = total_loss / test_batch_size

            if max_loss > total_loss:
                max_loss = total_loss
                print("models saved iter: " + str(iteration))
                torch.save(encoder, encoder_model)
                torch.save(guesser, guesser_model)
                torch.save(decoder, decoder_model)

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
