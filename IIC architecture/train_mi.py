from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch

from torch.autograd import Variable
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from IID_loss import IID_loss
from entropy_loss import entropy_balance_loss
from four_variate_mi import four_variate_IID_loss
from transform_utils import scale, rotate, random_erease, vertical_flip
import sys
EPS=sys.float_info.epsilon
from iid_net import IIDNet

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 256
EVAL_FREQ_DEFAULT = 400

FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def neighbours(convolutions, perimeter=3):
    size = convolutions.shape[0]
    inputs = []
    reception_field = 2 * perimeter + 1
    step = 3

    for i in range(3, size-3, step):
        for j in range(3, size-3, step):
            conv_loss_tensor = torch.zeros(reception_field, reception_field)

            for row in range(i-perimeter, i+perimeter+1):
                for col in range(j-perimeter, j+perimeter+1):
                    if row >= 0 and row < size:
                        if col >= 0 and col < size:
                            conv_loss_tensor[row - (i-perimeter), col - (j-perimeter)] = convolutions[row, col]

            flatten_input = torch.flatten(conv_loss_tensor)
            inputs.append(flatten_input)

    return inputs


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def forward_block(X, ids, colons, optimizers, train, to_tensor_size):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, to_tensor_size)

    images = x_tensor/255

    augments = {
        0: rotate(images, 20, BATCH_SIZE_DEFAULT),
        1: rotate(images, -20, BATCH_SIZE_DEFAULT),
        2: scale(images, BATCH_SIZE_DEFAULT),
        3: images
    }

    ids = np.random.choice(len(augments), size=4, replace=False)

    image_1 = augments[ids[0]]
    image_2 = augments[ids[1]]

    image_1 = image_1.to('cuda')
    image_2 = image_2.to('cuda')

    pred_1 = colons[0](image_1)
    pred_2 = colons[0](image_2)

    mean_pred = (pred_1 + pred_2) / 2
    mean_pred[(mean_pred < EPS).data] = EPS
    loss = entropy_balance_loss(mean_pred, 1)
    #loss = IID_loss(pred_1, pred_2)

    if train:
        optimizers[0].zero_grad()
        loss.backward(retain_graph=True)
        optimizers[0].step()

        # optimizers[1].zero_grad()
        # loss.backward(retain_graph=True)
        # optimizers[1].step()

    return pred_1, pred_2, loss


def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target[60000:]

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    colons = []

    optimizers = []
    colons_paths = []

    filepath = 'colons\\colon_' + str(0) + '.model'
    predictor_model = os.path.join(script_directory, filepath)
    colons_paths.append(predictor_model)

    #input = 5120
    input = 4608

    # c = Ensemble()
    # c.cuda()

    c = IIDNet(1, input)
    c.cuda()
    colons.append(c)

    # c2 = Colon(1, input)
    # c2.cuda()
    # colons.append(c2)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    # optimizer2 = torch.optim.Adam(c2.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer2)

    max_loss = 1999
    max_loss_iter = 0

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)

            print()
            print("iteration: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best loss: ", max_loss_iter,
                  ": ", max_loss)

            print_info(p1, p2, targets, test_ids)

            test_loss = mim.item()

            if max_loss > test_loss:
                max_loss = test_loss
                max_loss_iter = iteration
                measure_acc_augments(X_test, colons, targets)
                print("models saved iter: " + str(iteration))
                # for i in range(number_colons):
                #     torch.save(colons[i], colons_paths[i])

            print("test loss " + str(test_loss))
            print("")


def measure_acc_augments(X_test, colons, targets):
    print_dict = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "0": []}
    runs = len(X_test)//BATCH_SIZE_DEFAULT
    avg_loss = 0

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        optimizers = []
        p1, p2, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)

        avg_loss += mim.item()
        for i in range(p1.shape[0]):
            val, index = torch.max(p1[i], 0)
            val, index2 = torch.max(p2[i], 0)

            preds = [index.data.cpu().numpy(), index2.data.cpu().numpy()]
            preds = list(preds)
            preds = [int(x) for x in preds]
            # print(preds)
            verdict = most_frequent(preds)

            # print("verdict", verdict)
            # print("target", targets[test_ids[i]])
            # input()
            label = targets[test_ids[i]]
            if label == 10:
                label = 0

            print_dict[label].append(verdict)

    total_miss = 0
    for element in print_dict.keys():
        length = len(print_dict[element])
        misses = miss_classifications(print_dict[element])
        total_miss += misses

        print("cluster: ",
              element,
              ", most frequent: ",
              most_frequent(print_dict[element]),
              ", miss-classifications: ",
              misses,
              ", miss percentage: ",
              misses / length)


    print()
    print("AUGMENTS avg loss: ", avg_loss / runs,
          " miss: ", total_miss,
          " data: ", runs * BATCH_SIZE_DEFAULT,
          " miss percent: ", total_miss / (runs * BATCH_SIZE_DEFAULT))
    print()


def miss_classifications(cluster):
    mfe = most_frequent(cluster)
    missclassifications = 0
    for j in cluster:
        if j != mfe:
            missclassifications += 1

    return missclassifications

def most_frequent(List):
    return max(set(List), key=List.count)

def to_Tensor(X, batch_size=BATCH_SIZE_DEFAULT):
    X = np.reshape(X, (batch_size, 1, 28, 28))
    X = Variable(torch.FloatTensor(X))

    return X


def show_mnist(first_image):
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def print_info(p1, p2, targets, test_ids):
    print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
    for i in range(p1.shape[0]):
        if i == 10:
            print("")

        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + ", "

        label = targets[test_ids[i]]
        print_dict[label] += string

    for i in print_dict.keys():
        print(i, " : ", print_dict[i])


def main():
    """
    Main function
    """
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

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