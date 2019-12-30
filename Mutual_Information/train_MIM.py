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
from conv_net import ConvNet
from sklearn.datasets import fetch_openml
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import copy


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1000'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 64
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


def calc_all_distances(results):
    size = len(results)-1
    distances = 0
    count = 0
    for i in range(size):
        for j in range(i+1, size):
            distances += calc_distance(results[i], results[j])
            count += 1

    return distances / count


def calc_distance(out, out_2):
    abs_difference = torch.abs(out - out_2)
    eps = 1e-8
    information_loss = torch.log(1 - abs_difference + eps)
    mean = torch.mean(information_loss)

    return torch.abs(mean)


def forward_block(X, ids, y, model):
    ######## prepare input 1 for encoder 1 ######

    x_train = X[ids, :]
    x = get_data(x_train)

    ######## prepare input 2 for encoder 2 ######

    new_ids = np.random.choice(len(X), size=HALF_BATCH, replace=False)
    concat_ids = np.concatenate((ids[:HALF_BATCH], new_ids), axis=0)

    x_train_new = X[concat_ids, :]
    x_new = get_data(x_train_new)

    results = model.forward(x, x_new)

    ce_loss = nn.functional.binary_cross_entropy(results[0], y)
    distances_loss = calc_all_distances(results)

    coeff = 1.1
    loss = ce_loss + coeff * distances_loss
    return loss, results[0], distances_loss


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    # z = add_noise(X_train[2:3])
    # show_mnist(z[0])
    # z = add_noise(X_train[2:3])
    # show_mnist(z[0])
    # z = add_noise(X_train[2:3])
    # show_mnist(z[0])
    # z = add_noise(X_train[2:3])
    # show_mnist(z[0])
    # z = add_noise(X_train[2:3])
    # show_mnist(z[0])
    # z = add_noise(X_train[2:3])
    # show_mnist(z[0])
    # z = add_noise(X_train[2:3])
    # show_mnist(z[0])
    # z = add_noise(X_train[2:3])
    # show_mnist(z[0])
    # z = add_noise(X_train[2:3])
    # show_mnist(z[0])

    print(X_test.shape)

    input_dim = 256

    model = MutualInfoMetric()
    optimizer = torch.optim.Adam(model.parameters())

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'mi_encoder_1.model'
    encoder1_model = os.path.join(script_directory, filepath)

    filepath = 'mi_encoder_2.model'
    encoder2_model = os.path.join(script_directory, filepath)

    filepath = 'mi_discriminator.model'
    mi_discriminator_model = os.path.join(script_directory, filepath)

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
        model.train()

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        loss, mlp_out, distances = forward_block(X_train, ids, y_train_batch, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % EVAL_FREQ_DEFAULT == 0:
            model.eval()

            total_acc = 0
            total_loss = 0
            similarity_total = 0

            accuracies = []
            losses = []

            test_size = 9984
            for i in range(BATCH_SIZE_DEFAULT, test_size, BATCH_SIZE_DEFAULT):

                ids = np.array(range(i - BATCH_SIZE_DEFAULT, i))
                loss, mlp_out, distances = forward_block(X_test, ids, y_test_batch, model)

                total_loss += loss.item()
                similarity_total += distances.item()

                # TODO fix
                acc = accuracy(mlp_out, y_test_batch)
                total_acc += acc

            denom = test_size // BATCH_SIZE_DEFAULT
            total_acc = total_acc / denom
            total_loss = total_loss / denom
            similarity_total = similarity_total / denom
            accuracies.append(total_acc)
            losses.append(total_loss)

            if best_accuracy < total_acc:
                best_accuracy = total_acc
                print("models saved iter: " + str(iteration))
                torch.save(model.encoder, encoder1_model)
                torch.save(model.encoder_2, encoder2_model)
                torch.save(model.discriminator, mi_discriminator_model)

            print("total accuracy " + str(total_acc) + " total loss " + str(total_loss)+" similarity: "+str(similarity_total))

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
    x_rotate = rotate(x_train)

    data.append(x_original)
    data.append(x_noised)
    data.append(x_scaled)
    data.append(x_rotate)

    return data


def mutual_info_loss(original, augmented):
    """ Mutual information for joint histogram
    ...     """
    T = 1
    D = original.shape[1]
    original = torch.nn.functional.softmax(original / T, dim=1)
    augmented = torch.nn.functional.softmax(augmented / T, dim=1)

    P = (original.unsqueeze(2) * augmented.unsqueeze(1)).sum(dim=0)
    P = ((P + P.t()) / 2) / P.sum()
    Pi = P.sum(dim=1).view(D, 1).expand(D, D)
    Pj = P.sum(dim=0).view(1, D).expand(D, D)
    loss = (P * (torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum()

    return loss

    pxy = torch.div(semantic_representation, sum.view(2, 2, D))

    print(pxy.shape)
    print(pxy[0])
    input()
    # print()

    px = torch.sum(pxy, 1)  # marginal for x over y
    print(px.shape)
    print(px[0])
    input()
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def to_Tensor(X, batch_size=BATCH_SIZE_DEFAULT):
    X = np.reshape(X, (batch_size, 1, 28, 28))
    X = Variable(torch.FloatTensor(X))

    return X


def add_noise(X, batch_size=BATCH_SIZE_DEFAULT, max_noise_percentage=0.5):
    X_copy = copy.deepcopy(X)
    threshold = random.uniform(0, max_noise_percentage)

    for i in range(X_copy.shape[0]):
        nums = np.random.uniform(low=0, high=1, size=(X_copy[i].shape[0],))
        X_copy[i] = np.where(nums > threshold, X_copy[i], 0)

    return to_Tensor(X_copy, batch_size)


def rotate(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = to_Tensor(X_copy, batch_size)

    for i in range(X_copy.shape[0]):
        transformation = transforms.RandomRotation(20)
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def scale(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = to_Tensor(X_copy, batch_size)
    size = 22
    pad = 3

    if random.uniform(0,1) > 0.5:
        size = 18
        pad = 5

    for i in range(X_copy.shape[0]):
        transformation = transforms.Resize(size, interpolation=2)
        trans = transforms.Compose([transformation, transforms.Pad(pad), transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def augment(X):
    augment = [1, 2, 3]
    method = random.choice(augment)

    if method == 1:
        return add_noise(X)
    elif method == 2:
        return rotate(X)
    elif method == 3:
        return scale(X)


def show_mnist(first_image):
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def display_transformations(image):
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