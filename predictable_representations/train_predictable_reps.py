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
LEARNING_RATE_DEFAULT = 5e-7
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 1
EVAL_FREQ_DEFAULT = 1000

FLAGS = None


def calc_distance(out, y):
    abs_difference = torch.abs(out - y)
    abs_difference = torch.min(torch.ones(abs_difference.shape), abs_difference)
    abs_difference = torch.max(torch.zeros(abs_difference.shape), abs_difference)

    eps = 1e-8
    information_loss = torch.log(1 - abs_difference + eps)
    # print(information_loss)
    # input()
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


def sample(X):
    id = np.random.choice(len(X), size=1, replace=False)
    x_train = X[id]

    return x_train

def split_image_to_4(image):
    image_shape = image.shape

    image_a, image_b = torch.split(image, image_shape[2]//2, dim=2)

    image_1, image_2 = torch.split(image_a, image_shape[2]//2, dim=3)
    image_3, image_4 = torch.split(image_b, image_shape[2]//2, dim=3)
    # print(image_1.shape)
    # print(image_2.shape)
    # print(image_3.shape)
    # print(image_4.shape)
    return image_1, image_2, image_3, image_4


def encode_patches(i_1, i_2, i_3, i_4, encoder):
    encoded_1 = encoder.forward(i_1)
    encoded_2 = encoder.forward(i_2)
    encoded_3 = encoder.forward(i_3)
    encoded_4 = encoder.forward(i_4)

    flattened_1 = flatten(encoded_1)
    flattened_2 = flatten(encoded_2)
    flattened_3 = flatten(encoded_3)
    flattened_4 = flatten(encoded_4)

    return flattened_1, flattened_2, flattened_3, flattened_4

def guess_patches(flattened_1, flattened_2, flattened_3, flattened_4, guesser):
    input_1 = torch.cat([flattened_2, flattened_3, flattened_4], 1)
    input_2 = torch.cat([flattened_1, flattened_3, flattened_4], 1)
    input_3 = torch.cat([flattened_1, flattened_2, flattened_4], 1)
    input_4 = torch.cat([flattened_1, flattened_2, flattened_3], 1)

    guessed_1 = guesser(input_1)
    guessed_2 = guesser(input_2)
    guessed_3 = guesser(input_3)
    guessed_4 = guesser(input_4)

    return guessed_1, guessed_2, guessed_3, guessed_4

def forward_decoder(X, ids, encoder, guesser, decoder, optimizers, train):
    if train:
        decoder.train()
        guesser.train()
        encoder.train()
    else:
        decoder.eval()
        guesser.eval()
        encoder.eval()

    #### image 1 ###

    x_1 = sample(X)
    x_tensor_1 = to_Tensor(x_1, 1)
    image_1 = x_tensor_1 / 255

    i_1, i_2, i_3, i_4 = split_image_to_4(image_1)
    flattened_1, flattened_2, flattened_3, flattened_4 = encode_patches(i_1, i_2, i_3, i_4, encoder)
    guessed_1, guessed_2, guessed_3, guessed_4 = guess_patches(flattened_1, flattened_2, flattened_3, flattened_4, guesser)

    #### image 2 ###

    x_2 = sample(X)
    x_tensor_2 = to_Tensor(x_2, 1)
    image_2 = x_tensor_2 / 255

    print(flattened_1.shape)
    print(input_1.shape)
    print("guessed", guessed_1.shape)
    loss_1 = calc_distance(flattened_1, guessed_1)
    loss_2 = calc_distance(flattened_2, guessed_2)
    loss_3 = calc_distance(flattened_3, guessed_3)
    loss_4 = calc_distance(flattened_4, guessed_4)

    guessed_image = torch.cat([guessed_1, guessed_2, guessed_3, guessed_4], 1)
    print(guessed_image.shape)

    print(loss_1)
    print(loss_2)
    print(loss_3)
    print(loss_4)
    print("encoded image shape", encoded_image.shape)
    input()
    flatten_encoded_image = multi_filter_flatten(encoded_image)


    guessed_image = torch.zeros(encoded_image.shape)

    # Guess the convolved here
    inputs = []
    counter = 0
    for i in encoded_image[0]:
        filter_inputs = neighbours(i, 1)
        inputs.append(filter_inputs)
        guessed_value = guesser(filter_inputs[0])
        guessed_image[0][counter] = guessed_value

    losses = torch.zeros(len(inputs))


    for i in range(len(inputs)):
        counter = 0
        for j in inputs[i]:
            guess = guesser(j)
            # print(j)
            # print(j.shape)
            # print("guess", guess)
            # print("flatten conv: ", flatten_encoded_image[0][i][counter])
            # input()
            guess_loss = calc_distance(guess[0], flatten_encoded_image[0][i][counter])

            losses[i] = guess_loss
            counter += 1

    sum_guess_loss = torch.sum(losses)
    sum_guess_loss = torch.abs(sum_guess_loss)

    flattened_image = flatten(image)

    fully_flattened_encoded = flatten(flatten_encoded_image)

    flattened_guessed_image = flatten(guessed_image)
    res = decoder.forward(flattened_guessed_image)

    reconstrution_loss = calc_distance(res, flattened_image)

    total_loss = reconstrution_loss  # + sum_guess_loss

    if train:
        # encoder
        optimizers[0].zero_grad()
        total_loss.backward(retain_graph=True)
        optimizers[0].step()

        # guesser
        optimizers[1].zero_grad()
        total_loss.backward(retain_graph=True)
        optimizers[1].step()

        # decoder
        optimizers[2].zero_grad()
        reconstrution_loss.backward(retain_graph=True)
        optimizers[2].step()

    return total_loss.item(), reconstrution_loss.item(), sum_guess_loss.item(), res, flatten_encoded_image


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

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(encoder_optimizer)

    guesser_optimizer = torch.optim.Adam(guesser.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(guesser_optimizer)

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(decoder_optimizer)
    max_loss = 10000000

    for iteration in range(MAX_STEPS_DEFAULT):
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        loss_list, reconstruction_loss, guess_loss, res, flatten_convolved_image = forward_decoder(X_train, ids, encoder, guesser, decoder, optimizers, train)

        if iteration % 100 == 0:
            print("iteration: ", iteration)
            print("total loss: ", loss_list)
            print("guess loss: ", guess_loss)
            print("reconstruction_loss: ", reconstruction_loss)
            print("")

        if iteration % EVAL_FREQ_DEFAULT == 0:

            total_loss = 0

            test_batch_size = 500

            test_ids = np.random.choice(len(X_test), size=test_batch_size, replace=False)

            for c, i in enumerate(test_ids):
                if c % 100 == 0:
                    print("test iteration: "+str(c))
                loss_list, reconstruction_loss, guess_loss, res, flatten_convolved_image = forward_decoder(X_test, i, encoder, guesser, decoder, optimizers, False)

                #total_loss += statistics.mean(loss_list)
                total_loss += loss_list

            total_loss = total_loss / test_batch_size

            if max_loss > total_loss:
                max_loss = total_loss
                print("models saved iter: " + str(iteration))
                torch.save(encoder, encoder_model)
                torch.save(guesser, guesser_model)
                torch.save(decoder, decoder_model)

            print("total test loss " + str(total_loss))
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
