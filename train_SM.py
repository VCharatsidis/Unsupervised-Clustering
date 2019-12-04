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
from SimilarityMetric import SimilarityMetric
from torch.autograd import Variable
import matplotlib.pyplot as plt
from conv_net import ConvNet
from sklearn.datasets import fetch_openml
import torch.nn as nn

import random


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 20000
BATCH_SIZE_DEFAULT = 128
HALF_BATCH = BATCH_SIZE_DEFAULT // 2
EVAL_FREQ_DEFAULT = 400

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
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    input_dim = 256

    sm = SimilarityMetric()
    optimizer = torch.optim.Adam(sm.parameters())

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'encoder_1.model'
    encoder1_model = os.path.join(script_directory, filepath)

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'encoder_2.model'
    encoder2_model = os.path.join(script_directory, filepath)

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'discriminator.model'
    discriminator_model = os.path.join(script_directory, filepath)
    min_loss = -1000

    ones = torch.ones(HALF_BATCH)
    zeros = torch.zeros(HALF_BATCH)

    y_test_batch = torch.cat([ones, zeros], 0)
    y_test_batch = Variable(torch.FloatTensor(y_test_batch.float()))

    y_train_batch = torch.cat([ones, zeros], 0)
    y_train_batch = Variable(torch.FloatTensor(y_train_batch.float()))

    max_loss = 100

    for iteration in range(MAX_STEPS_DEFAULT):
        sm.train()

        ######## prepare input 1 for encoder 1 ######
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        X_train_clean = X_train[ids, :]
        threshold = random.uniform(0.4, 1)
        for i in range(X_train_clean.shape[0]):
            nums = np.random.uniform(low=0, high=1, size=(X_train_clean[i].shape[0],))
            X_train_clean[i] = np.where(nums > threshold, X_train_clean[i], 0)

        X_train_clean = np.reshape(X_train_clean, (BATCH_SIZE_DEFAULT, 1, 28, 28))
        X_train_clean = Variable(torch.FloatTensor(X_train_clean))

        ######## prepare input 2 for encoder 2 ######

        new_ids = np.random.choice(len(X_train), size=HALF_BATCH, replace=False)
        concat_ids = np.concatenate((ids[:HALF_BATCH], new_ids), axis=0)
        X_train_new = X_train[concat_ids, :]

        threshold = random.uniform(0.4, 1)
        for i in range(X_train_new.shape[0]):
            nums = np.random.uniform(low=0, high=1, size=(X_train_new[i].shape[0],))
            X_train_new[i] = np.where(nums > threshold, X_train_new[i], 0)

        X_train_new = np.reshape(X_train_new, (BATCH_SIZE_DEFAULT, 1, 28, 28))
        X_train_new = Variable(torch.FloatTensor(X_train_new))

        output = sm.forward(X_train_clean, X_train_new)

        loss = nn.functional.binary_cross_entropy(output, y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % EVAL_FREQ_DEFAULT == 0:
            sm.eval()

            total_acc = 0
            total_loss = 0

            accuracies = []
            losses = []

            test_size = 9984
            for i in range(BATCH_SIZE_DEFAULT, test_size, BATCH_SIZE_DEFAULT):

                ######## prepare input 1 for encoder 1 ######
                threshold = random.uniform(0.4, 1)
                ids = np.array(range(i - BATCH_SIZE_DEFAULT, i))
                X_test_clean = X_test[ids, :]

                for i in range(X_test_clean.shape[0]):
                    nums = np.random.uniform(low=0, high=1, size=(X_test_clean[i].shape[0],))
                    X_test_clean[i] = np.where(nums > threshold, X_test_clean[i], 0)

                X_test_clean = np.reshape(X_test_clean, (BATCH_SIZE_DEFAULT, 1, 28, 28))
                X_test_clean = Variable(torch.FloatTensor(X_test_clean))

                ######## prepare input 2 for encoder 2 ######
                threshold = random.uniform(0.4, 1)
                new_ids = np.random.choice(len(X_test), size=HALF_BATCH, replace=False)
                concat_ids = np.concatenate((ids[:HALF_BATCH], new_ids), axis=0)
                X_test_new = X_test[concat_ids, :]

                for i in range(X_test_new.shape[0]):
                    nums = np.random.uniform(low=0, high=1, size=(X_test_new[i].shape[0],))
                    X_test_new[i] = np.where(nums > threshold, X_test_new[i], 0)

                X_test_new = np.reshape(X_test_new, (BATCH_SIZE_DEFAULT, 1, 28, 28))
                X_test_new = Variable(torch.FloatTensor(X_test_new))

                output = sm.forward(X_test_clean, X_test_new)

                ####### calc losses ########

                loss_test = nn.functional.binary_cross_entropy(output, y_test_batch)
                total_loss += loss_test.item()

                acc = accuracy(output, y_test_batch)
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
                max_loss = total_loss
                print("models saved iter: " + str(iteration))
                torch.save(sm.encoder, encoder1_model)
                torch.save(sm.encoder2, encoder2_model)
                torch.save(sm.discriminator, discriminator_model)

            print("total accuracy " + str(total_acc) + " total loss " + str(total_loss))

    plt.plot(accuracies)
    plt.ylabel('accuracies')
    plt.show()

    plt.plot(losses)
    plt.ylabel('losses')
    plt.show()


def main():
    """
    Main function
    """

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