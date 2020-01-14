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
from SimilarityMetric.SimilarityMetric import SimilarityMetric
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import random

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 64
HALF_BATCH = BATCH_SIZE_DEFAULT // 2
EVAL_FREQ_DEFAULT = 300

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
    filepath = 'discriminator3.model'
    discriminator3_model = os.path.join(script_directory, filepath)

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'discriminator4.model'
    discriminator4_model = os.path.join(script_directory, filepath)

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'discriminator5.model'
    discriminator5_model = os.path.join(script_directory, filepath)

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'discriminator6.model'
    discriminator6_model = os.path.join(script_directory, filepath)

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'meta_discriminator.model'
    meta_discriminator_model = os.path.join(script_directory, filepath)

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
        X_train_clean = np.reshape(X_train_clean, (BATCH_SIZE_DEFAULT, 1, 28, 28))
        X_train_clean = Variable(torch.FloatTensor(X_train_clean))

        #rotate_image(X_train_clean[0])

        ######## prepare input 2 for encoder 2 ######

        new_ids = np.random.choice(len(X_train), size=HALF_BATCH, replace=False)
        concat_ids = np.concatenate((ids[:HALF_BATCH], new_ids), axis=0)

        X_train_new = X_train[concat_ids, :]
        X_train_new = transformations(X_train_new)

        output, out_inv = sm.forward(X_train_clean, X_train_new)

        loss = nn.functional.binary_cross_entropy(output, y_train_batch) + torch.mean(torch.abs(output-out_inv))

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
                ids = np.array(range(i - BATCH_SIZE_DEFAULT, i))
                X_test_clean = X_test[ids, :]

                X_test_clean = np.reshape(X_test_clean, (BATCH_SIZE_DEFAULT, 1, 28, 28))
                X_test_clean = Variable(torch.FloatTensor(X_test_clean))

                ######## prepare input 2 for encoder 2 ######

                new_ids = np.random.choice(len(X_test), size=HALF_BATCH, replace=False)
                concat_ids = np.concatenate((ids[:HALF_BATCH], new_ids), axis=0)

                X_test_new = X_test[concat_ids, :]
                X_test_new = transformations(X_test_new)

                output, out_inv = sm.forward(X_test_clean, X_test_new)

                ####### calc losses ########

                loss_test = nn.functional.binary_cross_entropy(output, y_test_batch) + torch.mean(torch.abs(output-out_inv))
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
                # torch.save(sm.discriminator_3, discriminator3_model)
                # torch.save(sm.discriminator_4, discriminator4_model)
                # torch.save(sm.discriminator_5, discriminator5_model)
                # torch.save(sm.discriminator_6, discriminator6_model)
                torch.save(sm.discriminator, meta_discriminator_model)

            print("total accuracy " + str(total_acc) + " total loss " + str(total_loss))

    plt.plot(accuracies)
    plt.ylabel('accuracies')
    plt.show()

    plt.plot(losses)
    plt.ylabel('losses')
    plt.show()


def transformations(X):
    noise = random.uniform(0, 1) > 0.6
    # #TODO
    noise = False
    if noise:
        threshold = random.uniform(0, 0.6)
        for i in range(X.shape[0]):
            nums = np.random.uniform(low=0, high=1, size=(X[i].shape[0],))
            X[i] = np.where(nums > threshold, X[i], 0)

    X = np.reshape(X, (BATCH_SIZE_DEFAULT, 1, 28, 28))
    X = Variable(torch.FloatTensor(X))

    if not noise:
        for i in range(X.shape[0]):
            rotate = random.uniform(0, 1) > 0.5
            # TODO
            rotate = False
            if rotate:
                transformation = transforms.RandomRotation(20)
                trans = transforms.Compose(
                    [transformation, transforms.ToTensor()])
            else:
                transformation = transforms.Resize(22, interpolation=1)
                trans = transforms.Compose(
                    [transformation, transforms.Pad(3), transforms.ToTensor()])

            a = F.to_pil_image(X[i])

            trans_image = trans(a)
            X[i] = trans_image

    return X


def show_mnist(first_image):
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def rotate_image(image):
    show_mnist(image)

    #transforms.RandomChoice

    #trans = transforms.Compose([transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)])
    #trans = transforms.Compose([transforms.RandomAffine(20)])
    #trans = transforms.Compose([transforms.Scale(20)])
    #trans = transforms.Compose([transforms.CenterCrop(23)])
    #trans = transforms.Resize(20, interpolation=2)
    trans = transforms.RandomRotation(30)
    #trans = transforms.Resize(20, interpolation=2)

    # transformation = transforms.RandomRotation(10)
    # trans = transforms.Compose(
    #     [transformation, transforms.ToTensor()])

    #trans = transforms.Compose([trans, transforms.Pad(4)])
    a = F.to_pil_image(image)

    trans_image = trans(a)

    pixels = trans_image.resize((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    input()

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