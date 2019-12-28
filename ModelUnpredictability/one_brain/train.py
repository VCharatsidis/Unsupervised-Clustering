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
from detached_conv import DetachedConvNet
from sklearn.datasets import fetch_openml
from predictor import Predictor
import statistics


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 4
HALF_BATCH = BATCH_SIZE_DEFAULT // 2
EVAL_FREQ_DEFAULT = 1000

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


def forward_block(X, ids, conv, predictor, optimizer, train):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train)

    convolutions = conv.forward(x_tensor)

    x = flatten(convolutions)

    size = x.shape[1]

    # One hot encoding buffer that you create out of the loop and just keep reusing
    i_one_hot = torch.FloatTensor(BATCH_SIZE_DEFAULT, size)

    total_loss = []
    total_distances = []

    for i in range(size):

        predictor.train()
        y = x[:, i]
        # y.unsqueeze_(0)
        # y.t_()

        # In your for loop
        i_one_hot.zero_()

        i_tensor = torch.LongTensor(BATCH_SIZE_DEFAULT, 1)
        i_tensor[:, 0] = i
        i_one_hot.scatter_(1, i_tensor, 1)

        x_reduced = torch.cat([x[:, 0:i], x[:, i + 1:]], 1)
        x_reduced = torch.cat([x_reduced, i_one_hot], 1)

        #print(x_reduced)
        res = predictor.forward(x_reduced.detach())

        distances = calc_distance(res, y)
        total_distances.append(distances)

        # if train:
        #     optimizer.zero_grad()
        #     distances.backward(retain_graph=True)
        #     optimizer.step()

        total_loss.append(distances.item())

    loss = 0
    for i in total_distances:
        loss += i

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss")
    print(loss)
    print(loss.shape)
    print(loss.item())

    return total_loss


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target

    X_train = mnist.data[:4]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    input_dim = 256

    conv = DetachedConvNet(1)
    predictor = Predictor(1351)
    optimizer = torch.optim.Adam(predictor.parameters())

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'detached_net.model'
    detached_model = os.path.join(script_directory, filepath)

    filepath = 'predictor.model'
    predictor_model = os.path.join(script_directory, filepath)

    ones = torch.ones(HALF_BATCH)
    zeros = torch.zeros(HALF_BATCH)

    y_test_batch = torch.cat([ones, zeros], 0)
    y_test_batch = Variable(torch.FloatTensor(y_test_batch.float()))

    y_train_batch = torch.cat([ones, zeros], 0)
    y_train_batch = Variable(torch.FloatTensor(y_train_batch.float()))

    max_loss = 100

    for iteration in range(MAX_STEPS_DEFAULT):
        print(iteration)
        predictor.train()

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        loss_list = forward_block(X_train, ids, conv, predictor, optimizer, train)

        print(loss_list)
        print(statistics.mean(loss_list))

        # if iteration % EVAL_FREQ_DEFAULT == 0:
        #     print(iteration)
        #     predictor.eval()
        #
        #     total_acc = 0
        #     total_loss = 0
        #     similarity_total = 0
        #
        #     accuracies = []
        #     # losses = []
        #
        #     test_size = 9984
        #     for i in range(BATCH_SIZE_DEFAULT, test_size, BATCH_SIZE_DEFAULT):
        #         ids = np.array(range(i - BATCH_SIZE_DEFAULT, i))
        #         loss_list = forward_block(X_test, ids, conv, predictor, optimizer, False)
        #
        #         total_loss += statistics.mean(loss_list)
        #
        #     denom = test_size // BATCH_SIZE_DEFAULT
        #     print(total_loss)
        #     total_acc = total_acc / denom
        #     total_loss = total_loss / denom
        #     similarity_total = similarity_total / denom
        #     accuracies.append(total_acc)
        #     # losses.append(total_loss)
        #
        #     if max_loss > total_loss:
        #         max_loss = total_loss
        #         print("models saved iter: " + str(iteration))
        #         torch.save(conv, detached_model)
        #         torch.save(predictor, predictor_model)
        #
        #     print("total accuracy " + str(total_acc) + " total loss " + str(total_loss)+" similarity: "+str(similarity_total))

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