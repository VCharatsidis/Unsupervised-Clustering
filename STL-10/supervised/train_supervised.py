from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch

from torch.autograd import Variable
import matplotlib.pyplot as plt
from stl10_input import read_all_images, read_labels
import torchvision.transforms.functional as F
from PIL import Image
from SupervisedNet import SupervisedNet
from stl_utils import rotate, scale, to_grayscale, random_erease, vertical_flip, horizontal_flip, sobel_filter_y, sobel_filter_x, sobel_total,center_crop, color_jitter
import random
import sys
from one_net_generator import OneNetGen, OneNetVAE
from torchvision.utils import make_grid
import matplotlib
from IID_loss import IID_loss
import torch.nn as nn
from torchvision import transforms
import torchvision


# Default constants
EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000

BATCH_SIZE_DEFAULT = 120
INPUT_NET = 4608
SIZE = 50
NETS = 1
DESCRIPTION = "Augments: 3 augments then 3 sobels x,y,total. Nets: "+str(NETS) +" net. Loss: total_loss = paired_losses - mean_probs_losses. Image size: " + str(SIZE)

EVAL_FREQ_DEFAULT = 200
NUMBER_CLASSES = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def encode(image, colons):
    pad = (96 - SIZE) // 2
    original_image = scale(image, SIZE, pad, BATCH_SIZE_DEFAULT)
    original_image = original_image[:, :, pad:96 - pad, pad:96 - pad]

    #show_image(original_image)

    original_image = original_image.to('cuda')
    preds = colons[0](original_image)

    return preds


def rgb2gray(rgb):

    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def show_gray(image_1):
    z = image_1
    print(z.shape)
    if len(list(z.size())) == 4:
        z = image_1.squeeze(1)

    pixels = z[0]
    plt.imshow(pixels, cmap='gray')
    plt.show()


def show_image(image_1):
    image_1 = torch.transpose(image_1, 1, 3)
    show_mnist(image_1[0], image_1[0].shape[1], image_1[0].shape[2])
    image_1 = torch.transpose(image_1, 1, 3)

    return image_1


def my_loss(preds_1, preds_2):
    product = preds_1 * preds_2
    product = product.mean(dim=0)  # * total_mean.detach()
    log_product = torch.log(product)
    loss = - log_product.mean(dim=0)

    return loss


def calc_distance(out, y):
    abs_difference = torch.abs(out - y)

    information_loss = torch.log(1 - abs_difference + EPS)

    return information_loss


def forward_block(X, ids, colons, optimizers, train, dictionary):
    images = X[ids, :]
    x_tensor = to_tensor(images, BATCH_SIZE_DEFAULT)

    images = x_tensor / 255.0

    preds = encode(images, colons)
    #print("preds shape", preds.shape)

    total_loss = torch.zeros([]).to('cuda')

    total_counter = 0
    for i in range(0, 10):
        y = torch.zeros([NUMBER_CLASSES]).to('cuda')
        y[i] = 1
        #print("zeros", y)
        sum_preds = torch.zeros([NUMBER_CLASSES]).to('cuda')
        product_preds = torch.ones([NUMBER_CLASSES]).to('cuda')
        cluster_ids = [id for id in ids if id in dictionary[i]]
        cluster_length = len(cluster_ids)
        #print("cluster length", cluster_length)
        if cluster_length ==0:
            continue

        for id in cluster_ids:
            index = np.where(ids == id)
            # print("sum preds", sum_preds.shape)
            # print("preds size", preds[index].squeeze().shape)
            # print("index", index)
            # print(preds[index].squeeze().shape)
            # print()
            product_preds *= preds[index].squeeze()
            sum_preds += preds[index].squeeze()
            total_counter += 1

        mean_preds = (sum_preds / cluster_length).to('cuda')
        #print("mean preds", mean_preds)

        #log_preds = -(y * torch.log(product_preds + EPS)).sum()
        H = - (mean_preds * torch.log(mean_preds + EPS)).sum()
        CE = - (y * torch.log(mean_preds + EPS)).sum()

        # print("H shape", H)
        # print("CE shape", CE)
        # print("total loss", total_loss)
        cluster_loss = H + CE
        total_loss += cluster_loss

    # print("total loss", total_loss)
    # print("total counter", total_counter)

    if train:
        optimizers[0].zero_grad()
        total_loss.backward(retain_graph=True)
        optimizers[0].step()

    return preds, total_loss


def entropies_loss(pred, coeff):
    return pred_entropy(pred) - coeff * batch_entropy(pred)


def pred_entropy(pred):
    H = - (pred * torch.log(pred)).sum(dim=1).mean(dim=0)

    return H


def batch_entropy(pred):
    batch_mean_preds = pred.mean(dim=0)
    H_batch = - (batch_mean_preds * torch.log(batch_mean_preds)).sum()

    return H_batch

def distance_loss(pred):
    mean_batch = pred.mean(dim=0)
    distance = torch.log(1 - torch.abs(mean_batch - 0.1))
    sum = -distance.sum()

    return sum


def save_image(original_image, iteration, name):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/{name}_iter_{iteration}.png", sample)


def accuracy(predictions, targets):
  predictions = predictions.cpu().detach().numpy()
  preds = np.argmax(predictions, 1)
  result = preds == targets
  sum = np.sum(result)
  accur = sum / float(targets.shape[0])

  return accur


def measure_acc_augments(X_test, colons, targets, test_dict):
    runs = len(X_test)//BATCH_SIZE_DEFAULT
    avg_loss = 0
    avg_accuracy = 0
    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        test_ids = np.array(test_ids)
        optimizers = []
        preds, mim = forward_block(X_test, test_ids, colons, optimizers, False, test_dict)

        avg_accuracy += accuracy(preds, targets[test_ids])
        avg_loss += mim.item()

    print()
    print("AUGMENTS avg loss: ", avg_loss / runs, " avg accuracy: ", avg_accuracy / runs)
    print()

    return avg_loss/runs, avg_accuracy / runs


def miss_classifications(cluster):
    mfe = most_frequent(cluster)
    missclassifications = 0
    for j in cluster:
        if j != mfe:
            missclassifications += 1

    return missclassifications

def most_frequent(List):
    return max(set(List), key=List.count)

def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():
    #
    fileName = "..\\data\\stl10_binary\\train_X.bin"
    X_train = read_all_images(fileName)

    train_y_File = "..\\data\\stl10_binary\\train_y.bin"
    y_train = read_labels(train_y_File)

    testFile = "..\\data\\stl10_binary\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "..\\data\\stl10_binary\\test_y.bin"
    targets = read_labels(test_y_File)

    # mnist = fetch_openml('mnist_784', version=1, cache=True)
    # targets = mnist.target[60000:]
    #
    # X_train = mnist.data[:60000]
    # X_test = mnist.data[60000:]

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    colons = []

    optimizers = []
    colons_paths = []

    filepath = 'encoders\\encoder_' + str(0) + '.model'
    predictor_model = os.path.join(script_directory, filepath)
    colons_paths.append(predictor_model)

    c = SupervisedNet(3, INPUT_NET)
    c = c.cuda()
    colons.append(c)
    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    max_loss = 1999
    max_loss_iter = 0
    most_clusters = 0
    most_clusters_iter = 0

    print(colons[0])

    labels_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    for idx, i in enumerate(y_train):
        if i == 10:
            i = 0
        labels_dict[i].append(idx)

    test_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    for idx, i in enumerate(targets):
        if i == 10:
            i = 0
        test_dict[i].append(idx)

    for iteration in range(MAX_STEPS_DEFAULT):
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        preds, mim = forward_block(X_train, ids, colons, optimizers, train, labels_dict)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            print()
            print("iteration: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best loss: ", max_loss_iter,
                  ": ", max_loss)
            print("description: ", DESCRIPTION)

            loss, clusters = measure_acc_augments(X_test, colons, targets, test_dict)

            if max_loss > loss:
                max_loss = loss
                max_loss_iter = iteration

                print("models saved iter: " + str(iteration))


def to_tensor(X, batch_size=BATCH_SIZE_DEFAULT):
    #X = np.reshape(X, (batch_size, 1, 96, 96))
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


def show_mnist(first_image, w, h):
    #pixels = first_image.reshape((w, h))
    pixels = first_image
    plt.imshow(pixels)
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