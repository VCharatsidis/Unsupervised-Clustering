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


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 1
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
    predictions = predictions.flatten()
    targets = targets.detach().numpy()

    preds = np.round(predictions)
    result = preds == targets

    sum = np.sum(result)
    accuracy = sum / float(targets.shape[0])

    return accuracy


def calc_distance(out, out_2):
    abs_difference = torch.abs(out - out_2)
    information_loss = torch.log(1 - abs_difference)
    mean = torch.mean(information_loss)

    return torch.abs(mean)


def flatten(out):
    return out.view(out.shape[0], -1)


def forward_block(X, ids, conv, predictor):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train)
    convolutions = conv.forward(x_tensor)
    x = flatten(convolutions)

    size = x.shape[1]

    loss = 0

    for i in range(size):
        y = x[:, i]
        x_reduced = torch.cat([x[:, 0:i], x[:, i + 1:]], 1)

        res = predictor.forward(x_reduced.detach())

        loss += calc_distance(res[0], y)

    return loss


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    input_dim = 256

    conv = DetachedConvNet(1)
    predictor = Predictor(168)
    optimizer = torch.optim.Adam(predictor.parameters())

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'detached_net.model'
    encoder1_model = os.path.join(script_directory, filepath)

    filepath = 'predictor.model'
    encoder1_model = os.path.join(script_directory, filepath)

    ones = torch.ones(HALF_BATCH)
    zeros = torch.zeros(HALF_BATCH)

    y_test_batch = torch.cat([ones, zeros], 0)
    y_test_batch = Variable(torch.FloatTensor(y_test_batch.float()))

    y_train_batch = torch.cat([ones, zeros], 0)
    y_train_batch = Variable(torch.FloatTensor(y_train_batch.float()))

    max_loss = 100
    saturation = 100
    best_accuracy = 0

    for iteration in range(MAX_STEPS_DEFAULT):
        conv.train()

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        loss = forward_block(X_train, ids, conv, predictor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % EVAL_FREQ_DEFAULT == 0:
            conv.eval()

            total_acc = 0
            total_loss = 0
            similarity_total = 0

            accuracies = []
            losses = []

            test_size = 9984
            for i in range(BATCH_SIZE_DEFAULT, test_size, BATCH_SIZE_DEFAULT):

                ids = np.array(range(i - BATCH_SIZE_DEFAULT, i))
                loss = forward_block(X_test, ids, conv, predictor)

                total_loss += loss.item()


            denom = test_size // BATCH_SIZE_DEFAULT
            total_acc = total_acc / denom
            total_loss = total_loss / denom
            similarity_total = similarity_total / denom
            accuracies.append(total_acc)
            losses.append(total_loss)

            if best_accuracy < total_acc:
                best_accuracy = total_acc
                print("models saved iter: " + str(iteration))
                torch.save(conv.encoder, encoder1_model)

            print("total accuracy " + str(total_acc) + " total loss " + str(total_loss)+" similarity: "+str(similarity_total))

    plt.plot(accuracies)
    plt.ylabel('accuracies')
    plt.show()

    plt.plot(losses)
    plt.ylabel('losses')
    plt.show()



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