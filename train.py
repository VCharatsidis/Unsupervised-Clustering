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
from datasets.bmnist import bmnist

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 6000
BATCH_SIZE_DEFAULT = 4
HALF_BATCH = BATCH_SIZE_DEFAULT // 2
EVAL_FREQ_DEFAULT = 200

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
    preds = np.argmax(predictions, 1)
    result = preds == targets
    sum = np.sum(result)
    accuracy = sum / float(targets.shape[0])

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    """

    X_train = _read_raw_image_file('data\\raw\\binarized_mnist_train.amat')
    X_test = _read_raw_image_file('data\\raw\\binarized_mnist_valid.amat')

    print(X_train[0].shape)

    input_dim = 10
    discriminator = MLP(input_dim)

    encoder = ConvNet(1)

    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=LEARNING_RATE_DEFAULT, momentum=0.9)
    encoder_optimizer = torch.optim.Adam(encoder.parameters())

    loss_fn = nn.CrossEntropyLoss()

    accuracies = []
    losses = []

    for iteration in range(MAX_STEPS_DEFAULT):
        discriminator.train()
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        X_train_batch = X_train[ids, :]
        X_train_batch = np.expand_dims(X_train_batch, axis=0)
        X_train_batch = X_train_batch.transpose(1, 0, 2, 3)
        X_train_batch = Variable(torch.IntTensor(X_train_batch).float())

        encoder_output = encoder.forward(X_train_batch)

        false_source = list(range(HALF_BATCH, BATCH_SIZE_DEFAULT))
        true_source_ids = np.array(list(range(0, HALF_BATCH)))

        permutation = np.random.permutation(false_source)
        new_list_idx = np.concatenate((true_source_ids, permutation), axis=0)

        X_train_batch = np.reshape(X_train_batch, (BATCH_SIZE_DEFAULT, -1))

        perm = torch.LongTensor(new_list_idx)
        X_train_batch = X_train_batch[perm, :]

        discriminator_input = torch.cat([X_train_batch, encoder_output], 1)

        ones = torch.ones(HALF_BATCH)
        zeros = torch.zeros(HALF_BATCH)

        y_train_batch = torch.cat([ones, zeros], 0)
        y_train_batch = Variable(torch.LongTensor(y_train_batch).long())

        discriminator_output = discriminator.forward(discriminator_input)

        loss_discriminator = loss_fn(discriminator_output, y_train_batch)
        discriminator_optimizer.zero_grad()
        loss_discriminator.backward()
        discriminator_optimizer.step()

        loss_encoder = loss_fn(discriminator_output, y_train_batch)
        encoder_optimizer.zero_grad()
        loss_encoder.backward()
        encoder_optimizer.step()

        if iteration % EVAL_FREQ_DEFAULT == 0:
            discriminator.eval()
            encoder.eval()
            total_acc = 0
            total_loss = 0

            for i in range(BATCH_SIZE_DEFAULT, len(X_test) + BATCH_SIZE_DEFAULT, BATCH_SIZE_DEFAULT):
                ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)

                X_test_batch = X_test[ids, :]
                X_test_batch = np.expand_dims(X_test_batch, axis=0)
                X_test_batch = X_test_batch.transpose(1, 0, 2, 3)
                X_test_batch = Variable(torch.IntTensor(X_test_batch).float())

                encoder_output = encoder.forward(X_test_batch)

                false_source = list(range(HALF_BATCH, BATCH_SIZE_DEFAULT))
                true_source_ids = np.array(list(range(0, HALF_BATCH)))

                permutation = np.random.permutation(false_source)
                new_list_idx = np.concatenate((true_source_ids, permutation), axis=0)

                X_test_batch = np.reshape(X_test_batch, (BATCH_SIZE_DEFAULT, -1))

                perm = torch.LongTensor(new_list_idx)
                X_test_batch = X_test_batch[perm, :]

                discriminator_input = torch.cat([X_test_batch, encoder_output], 1)

                ones = torch.ones(HALF_BATCH)
                zeros = torch.zeros(HALF_BATCH)

                y_test_batch = torch.cat([ones, zeros], 0)
                y_test_batch = Variable(torch.LongTensor(y_test_batch).long())

                discriminator_output = discriminator.forward(discriminator_input)
                print(discriminator_output)
                input()

            denom = len(X_test) / BATCH_SIZE_DEFAULT
            total_acc = total_acc / denom
            total_loss = total_loss / denom
            accuracies.append(total_acc)
            losses.append(total_loss)

            print("total accuracy " + str(total_acc) + " total loss " + str(total_loss))

    plt.plot(accuracies)
    plt.ylabel('accuracies')
    plt.show()

    plt.plot(losses)
    plt.ylabel('losses')
    plt.show()


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