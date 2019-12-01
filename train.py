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
from discriminator import MLP
import torch.nn as nn
#import cifar10_utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
from conv_net import ConvNet
from sklearn.datasets import fetch_openml

import random


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 6000
BATCH_SIZE_DEFAULT = 128
HALF_BATCH = BATCH_SIZE_DEFAULT // 2
EVAL_FREQ_DEFAULT = 500

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    """

    predictions = predictions.detach().numpy()
    predictions = predictions.flatten()
    targets = targets.detach().numpy()

    preds = np.round(predictions)
    result = preds == targets

    sum = np.sum(result)
    accuracy = sum / float(targets.shape[0])

    return accuracy

def train():
    """
    Performs training and evaluation of MLP model.

    """
    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'encoder.model'
    encoder_model = os.path.join(script_directory, filepath)

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'decoder.model'
    discriminator_model = os.path.join(script_directory, filepath)

    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    input_dim = 256
    discriminator = MLP(input_dim)

    encoder = ConvNet(1)

    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9)
    encoder_optimizer = torch.optim.Adam(encoder.parameters())

    ones = torch.ones(HALF_BATCH)
    zeros = torch.zeros(HALF_BATCH)

    y_test_batch = torch.cat([ones, zeros], 0)
    y_test_batch = Variable(torch.FloatTensor(y_test_batch.float()))

    y_train_batch = torch.cat([ones, zeros], 0)
    y_train_batch = Variable(torch.FloatTensor(y_train_batch.float()))

    max_loss = 100
    threshold = 0.75

    for iteration in range(MAX_STEPS_DEFAULT):
        discriminator.train()
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        X_train_clean = X_train[ids, :]

        pixels = X_train_clean[0].reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()

        for i in range(X_train_clean.shape[0]):
            nums = np.random.uniform(low=0, high=1, size=(X_train_clean[i].shape[0],))
            X_train_clean[i] = np.where(nums > threshold, X_train_clean[i], 0)

        X_train_clean = np.reshape(X_train_clean, (128, 1, 28, 28))
        #X_train_clean = np.expand_dims(X_train_clean, axis=0)
        #X_train_clean = X_train_clean.transpose(1, 0, 2, 3)
        print(X_train_clean[0])
        print(X_train_clean[0].shape)

        pixels = X_train_clean[0][0]
        plt.imshow(pixels, cmap='gray')
        plt.show()
        input()
        #X_train_clean = X_train_clean.transpose(1, 0, 2, 3)
        X_train_clean = Variable(torch.IntTensor(X_train_clean).float())

        encoder_output = encoder.forward(X_train_clean)

        ####### new ids ############

        new_ids = np.random.choice(len(X_train), size=HALF_BATCH, replace=False)
        concat_ids = np.concatenate((ids[:HALF_BATCH], new_ids), axis=0)
        X_train_new = X_train[concat_ids, :]

        for i in range(X_train_new.shape[0]):
            nums = np.random.uniform(low=0, high=1, size=(X_train_new[i].shape[0],))
            X_train_new[i] = np.where(nums > threshold, X_train_new[i], 0)

        X_train_new = np.expand_dims(X_train_new, axis=0)
        X_train_new = X_train_new.transpose(1, 0, 2, 3)
        X_train_new = Variable(torch.IntTensor(X_train_new).float())

        encoder_output_contrast = encoder.forward(X_train_new)

        discriminator_input = torch.cat([encoder_output, encoder_output_contrast], 1)
        discriminator_output = discriminator.forward(discriminator_input)

        loss_encoder = nn.functional.binary_cross_entropy(discriminator_output, y_train_batch)
        encoder_optimizer.zero_grad()
        loss_encoder.backward(retain_graph=True)
        encoder_optimizer.step()

        loss_discriminator = nn.functional.binary_cross_entropy(discriminator_output, y_train_batch)
        discriminator_optimizer.zero_grad()
        loss_discriminator.backward()
        discriminator_optimizer.step()

        if iteration % EVAL_FREQ_DEFAULT == 0:
            discriminator.eval()
            encoder.eval()
            total_acc = 0
            total_loss = 0

            accuracies = []
            losses = []

            test_size = 9984
            for i in range(BATCH_SIZE_DEFAULT, test_size, BATCH_SIZE_DEFAULT):
                ids = np.array(range(i - BATCH_SIZE_DEFAULT, i))

                X_test_clean = X_test[ids, :]

                X_test_batch = [[[noise_pixel(pixel) for pixel in row] for row in image] for image in X_test_clean]
                X_test_batch = np.expand_dims(X_test_batch, axis=0)
                X_test_batch = X_test_batch.transpose(1, 0, 2, 3)
                X_test_batch = Variable(torch.IntTensor(X_test_batch).float())

                encoder_output = encoder.forward(X_test_batch)

                ####### new ids ############

                new_ids = np.random.choice(len(X_test), size=HALF_BATCH, replace=False)
                concat_ids = np.concatenate((ids[:HALF_BATCH], new_ids), axis=0)
                X_test_new = X_test[concat_ids, :]
                X_test_batch = [[[noise_pixel(pixel) for pixel in row] for row in image] for image in X_test_new]
                X_test_batch = np.expand_dims(X_test_batch, axis=0)
                X_test_batch = X_test_batch.transpose(1, 0, 2, 3)
                X_test_batch = Variable(torch.IntTensor(X_test_batch).float())

                ####### Discriminate #######

                encoder_output_contrast = encoder.forward(X_test_batch)
                discriminator_input = torch.cat([encoder_output, encoder_output_contrast], 1)

                ####### calc losses ########

                discriminator_output = discriminator.forward(discriminator_input)

                loss_discriminator = nn.functional.binary_cross_entropy(discriminator_output, y_test_batch)
                total_loss += loss_discriminator.item()

                discriminator_output = discriminator.forward(discriminator_input)
                acc = accuracy(discriminator_output, y_test_batch)
                total_acc += acc

                # print(i)
                # print("accuracy discriminator: " + str(acc) + " loss discriminator: " + str(loss_discriminator.item()))
                # print(acc)

            denom = test_size // BATCH_SIZE_DEFAULT  # len(X_test) / BATCH_SIZE_DEFAULT
            total_acc = total_acc / denom
            total_loss = total_loss / denom
            accuracies.append(total_acc)
            losses.append(total_loss)

            if max_loss > total_loss:
                print("models saved iter: " + str(iteration))
                torch.save(encoder, encoder_model)
                torch.save(discriminator, discriminator_model)

            print("total accuracy " + str(total_acc) + " total loss " + str(total_loss))

    plt.plot(accuracies)
    plt.ylabel('accuracies')
    plt.show()

    plt.plot(losses)
    plt.ylabel('losses')
    plt.show()


def noise_pixel(pixel_value):
    if pixel_value == 0:
        return pixel_value

    if random.uniform(0, 1) < 0.75:
        return 0
    else:
        return pixel_value


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def _read_raw_image_file(path):
    raw_file = path
    all_images = []
    with open(raw_file) as f:
        for line in f:
            im = [int(x) for x in line.strip().split()]
            assert len(im) == 28 ** 2
            all_images.append(im)
    return torch.from_numpy(np.array(all_images)).view(-1, 28, 28)


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

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
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()