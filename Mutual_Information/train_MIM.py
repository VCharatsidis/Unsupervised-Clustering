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
from MutualInfoMetric import MutualInfoMetric
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import copy
import statistics
from RandomErase import RandomErasing
from torchvision.utils import make_grid


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 1e-5
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 1
HALF_BATCH = BATCH_SIZE_DEFAULT // 2
EVAL_FREQ_DEFAULT = 200


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


def sample(X):
    id = np.random.choice(len(X), size=1, replace=False)
    x_train = X[id]

    return x_train


def forward_block(X, y, model):

    sample_1 = sample(X)
    x1 = get_data(sample_1)

    sample_2 = sample(X)
    x2 = get_data(sample_2)

    # sample_3 = sample(X)
    # x3 = get_data(sample_3)
    #
    # sample_4 = sample(X)
    # x4 = get_data(sample_4)

    results, KL_1, KL_2, d, d1 = model.forward(x1, x2)
    # print("results")
    # print(results.shape)
    # print(results)
    # print("ÿ")
    # print(y.shape)
    # print(y)

    # print(results)
    # print(y)

    mean_1 = torch.sum(KL_1)
    mean_2 = torch.sum(KL_2)

    loss = nn.functional.binary_cross_entropy(results, y) + mean_1 + mean_2

    return loss, results, KL_1, KL_2, d, d1


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    model = MutualInfoMetric()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_DEFAULT)

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'mi_encoder_1.model'
    encoder1_model = os.path.join(script_directory, filepath)

    filepath = 'mi_encoder_2.model'
    encoder2_model = os.path.join(script_directory, filepath)

    filepath = 'mi_discriminator.model'
    mi_discriminator_model = os.path.join(script_directory, filepath)

    ones = torch.ones(98)
    zeros = torch.zeros(98)

    y_test_batch = torch.cat([ones, zeros], 0)
    y_test_batch = Variable(torch.FloatTensor(y_test_batch.float()))

    y_train_batch = torch.cat([ones, zeros], 0)
    y_train_batch = Variable(torch.FloatTensor(y_train_batch.float()))

    max_loss = 10000
    saturation = 100
    best_accuracy = 0
    train_accs = []

    for iteration in range(MAX_STEPS_DEFAULT):

        model.train()

        loss1, mlp_out1, k1, k2, _, _ = forward_block(X_train, y_train_batch, model)
        train_accuracy = accuracy(mlp_out1, y_test_batch)

        loss = loss1
        batch = 64

        for i in range(batch-1):
            loss_x, mlp_out_x, _, _, _, _ = forward_block(X_train, y_train_batch, model)

            loss += loss_x
            train_accuracy += accuracy(mlp_out_x, y_test_batch)

        loss /= batch
        train_accuracy /= batch
        train_accs.append(train_accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % EVAL_FREQ_DEFAULT == 0:
            if iteration > 1000:
                kernels = model.encoder.encoder[0].weight.detach().clone()

                kernels = kernels - kernels.min()
                kernels = kernels / kernels.max()
                img = make_grid(kernels)
                print(img.shape)
                plt.imshow(img.permute(1, 2, 0))
                plt.show()

            print(k1)

            print("train accuracies: ", statistics.mean(train_accs))
            train_accs = []

            #model.eval()

            total_acc = 0
            total_loss = 0
            similarity_total = 0

            accuracies = []
            losses = []

            test_size = 100
            with torch.no_grad():
                for i in range(test_size):

                    loss, mlp_out, _, _ , _, _= forward_block(X_test, y_test_batch, model)

                    total_loss += loss.item()

                    # TODO fix
                    acc = accuracy(mlp_out, y_test_batch)

                    if i == 5:
                        predictions = mlp_out.detach().numpy()
                        predictions = predictions.flatten()

                        targets = y_test_batch.detach().numpy()
                        print("predictions: ", predictions)
                        preds = np.round(predictions)
                        print("round preds: ", preds)
                        result = preds == targets

                        sum = np.sum(result)
                        print("sum: ", sum)
                        a = sum / float(targets.shape[0])
                        print("accuracy: ", a)

                    total_acc += acc

            denom = test_size // BATCH_SIZE_DEFAULT
            total_acc = total_acc / denom
            total_loss = total_loss / denom

            accuracies.append(total_acc)
            losses.append(total_loss)

            if best_accuracy < total_acc:
                best_accuracy = total_acc
                print("models saved iter: " + str(iteration))
                torch.save(model.encoder, encoder1_model)
                #torch.save(model.encoder_2, encoder2_model)
                torch.save(model.discriminator, mi_discriminator_model)

            print("total accuracy " + str(total_acc) + " total loss " + str(total_loss))

    plt.plot(accuracies)
    plt.ylabel('accuracies')
    plt.show()

    plt.plot(losses)
    plt.ylabel('losses')
    plt.show()


def get_data(x_train):
    data = []

    x_original = to_Tensor(x_train)
    x_noised = add_noise(x_train)
    x_scaled = scale(x_train)
    x_rotate = rotate(x_train, 25)
    x_rotate2 = rotate(x_train, -25)
    x_erase = random_erease(x_train)
    x_erase2 = random_erease(x_train)

    x_original = x_original.to('cuda')
    x_noised = x_noised.to('cuda')
    x_scaled = x_scaled.to('cuda')
    x_rotate = x_rotate.to('cuda')
    x_rotate2 = x_rotate2.to('cuda')
    x_erase = x_erase.to('cuda')
    x_erase2 = x_erase2.to('cuda')

    # show_mnist(x_original)
    # show_mnist(x_noised)
    # show_mnist(x_scaled)
    # show_mnist(x_rotate)
    # show_mnist(x_rotate2)
    # show_mnist(x_erase)
    # show_mnist(x_erase2)

    data.append(x_original)
    data.append(x_noised)
    data.append(x_scaled)
    data.append(x_rotate)
    data.append(x_rotate2)
    data.append(x_erase)
    data.append(x_erase2)

    return data


def to_Tensor(X, batch_size=BATCH_SIZE_DEFAULT):
    X = np.reshape(X, (batch_size, 1, 28, 28))
    X = Variable(torch.FloatTensor(X))

    return X


def add_noise(X, batch_size=BATCH_SIZE_DEFAULT, max_noise_percentage=0.5):
    X_copy = copy.deepcopy(X)
    threshold = random.uniform(0.4, max_noise_percentage)

    for i in range(X_copy.shape[0]):
        nums = np.random.uniform(low=0, high=1, size=(X_copy[i].shape[0],))
        X_copy[i] = np.where(nums > threshold, X_copy[i], 0)

    return to_Tensor(X_copy, batch_size)


def rotate(X, degrees, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = to_Tensor(X_copy, batch_size)

    for i in range(X_copy.shape[0]):
        transformation = transforms.RandomRotation(degrees=[degrees, degrees])
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def scale(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = to_Tensor(X_copy, batch_size)
    size = 20
    pad = 4

    # if random.uniform(0, 1) > 0.5:
    #     size = 20
    #     pad = 4

    for i in range(X_copy.shape[0]):
        transformation = transforms.Resize(size, interpolation=2)
        trans = transforms.Compose([transformation, transforms.Pad(pad), transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def random_erease(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = to_Tensor(X_copy, batch_size)

    for i in range(X_copy.shape[0]):
        transformation = RandomErasing()
        trans = transforms.Compose([transforms.ToTensor(), transformation])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


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