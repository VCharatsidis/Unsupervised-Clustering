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
from one_net_entropy import OneNetEntropy
from stl_utils import rotate, scale, to_grayscale, random_erease, vertical_flip, horizontal_flip, sobel_filter_y, sobel_filter_x, sobel_total,center_crop
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

BATCH_SIZE_DEFAULT = 62
INPUT_NET = 8192
SIZE = 40
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


def encode_4_patches(image, colons, optimizers, train,
                     p1=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p2=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p3=torch.zeros([BATCH_SIZE_DEFAULT, 10])):
    image /= 255

    # size = 40
    # pad = (96 - size) // 2
    # lydia = image[:, :, pad:96 - pad, pad:96 - pad]
    # show_gray(lydia)

    #show_gray(image)

    pad = (96-SIZE)//2
    original_image = scale(image, SIZE, pad, BATCH_SIZE_DEFAULT)
    #show_gray(original_image)

    original_image = original_image[:, :, pad:96-pad, pad:96-pad]
    #original_image = sobel_total(original_image, BATCH_SIZE_DEFAULT)

    augments = {0: horizontal_flip(original_image, BATCH_SIZE_DEFAULT),
                1: original_image,
                2: vertical_flip(original_image, BATCH_SIZE_DEFAULT),
                3: scale(original_image, SIZE-8, 4, BATCH_SIZE_DEFAULT),
                4: rotate(original_image, 20, BATCH_SIZE_DEFAULT),
                5: rotate(original_image, -20, BATCH_SIZE_DEFAULT),
                6: center_crop(image, SIZE, BATCH_SIZE_DEFAULT),
                7: sobel_filter_x(original_image, BATCH_SIZE_DEFAULT),
                8: sobel_filter_y(original_image, BATCH_SIZE_DEFAULT),
                9: sobel_total(original_image, BATCH_SIZE_DEFAULT)
                }

    ids = np.random.choice(len(augments), size=3, replace=False)

    image_1 = augments[ids[0]]
    image_2 = augments[ids[1]]

    # show_gray(image_1)
    # show_gray(image_2)

    p1 = p1.cuda()
    p2 = p2.cuda()
    p3 = p3.cuda()

    #net_id = np.random.choice(len(colons), size=4, replace=False)

    nets = [0, 0, 0]
    if NETS == 3:
        nets = [0, 1, 2]

    out_11, out_12, out_13, out_14, out_15, preds_1 = make_pred(image_1, colons, nets[0], p2, p3)
    # print("out_11", out_11.shape)
    # print("out_12", out_12.shape)
    # print("out_13", out_13.shape)
    # print("out_14", out_14.shape)
    # print("out_15", out_15.shape)
    # print("out_16", out_16.shape)
    # print("out_17", preds_1.shape)

    out_21, out_22, out_23, out_24, out_25, preds_2 = make_pred(image_2, colons, nets[1], p1, p3)

    original_image_cuda = original_image.to("cuda")
    image_3 = colons[-1](original_image_cuda, preds_1.detach()).reshape(BATCH_SIZE_DEFAULT, SIZE, SIZE).unsqueeze(1)
    out_31, out_32, out_33, out_34, out_35, preds_3 = colons[nets[2]](image_3, p1, p2)

    l11 = balance_entropy_loss(out_11)
    l12 = balance_entropy_loss(out_12)
    l13 = balance_entropy_loss(out_13)
    l14 = balance_entropy_loss(out_14)
    l15 = balance_entropy_loss(out_15)

    w_loss_1 = l15 #+ l11 + l12 + l13 + l14

    l21 = balance_entropy_loss(out_21)
    l22 = balance_entropy_loss(out_22)
    l23 = balance_entropy_loss(out_23)
    l24 = balance_entropy_loss(out_24)
    l25 = balance_entropy_loss(out_25)

    w_loss_2 = l25 #+ l21 + l22 + l23 + l24

    l31 = balance_entropy_loss(out_31)
    l32 = balance_entropy_loss(out_32)
    l33 = balance_entropy_loss(out_33)
    l34 = balance_entropy_loss(out_34)
    l35 = balance_entropy_loss(out_35)

    w_loss_3 = l35 #+ l31 + l32 + l33 + l34

    m_preds = (preds_1 + preds_2 + preds_3) / 3

    H = pred_entropy(m_preds)

    batch_pred_1 = batch_entropy(preds_1)
    batch_pred_2 = batch_entropy(preds_2)
    batch_pred_3 = batch_entropy(preds_3)

    w_loss = w_loss_1 + w_loss_2 + w_loss_3
    preds_loss = H - batch_pred_1 - batch_pred_2 - batch_pred_3

    total_loss = preds_loss + w_loss

    if train:
        # torch.autograd.set_detect_anomaly(True)

        optimizers[0].zero_grad()
        optimizers[-1].zero_grad()
        total_loss.backward(retain_graph=True)
        optimizers[0].step()
        optimizers[-1].step()

    return preds_1, preds_2, preds_3, image_3, total_loss


def balance_entropy_loss(x_raw):
    x = torch.flatten(x_raw, 1)

   # x[(x < EPS).data] = EPS

    H = - (x * torch.log(x+ EPS)).mean(dim=1).mean(dim=0)
    x_batch = x.mean(dim=0)
    #x_batch[(x_batch < EPS).data] = EPS

    H_batch = - (x_batch * torch.log(x_batch+ EPS)).mean()

    return H - H_batch


def make_pred(image, colons, net, p1, p2):
    image = image.to('cuda')
    out_1, out_2, out_3, out_4, out_5, out_6 = colons[net](image, p1, p2)
    del image
    torch.cuda.empty_cache()

    return out_1, out_2, out_3, out_4, out_5, out_6


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


def forward_block(X, ids, colons, optimizers, train, to_tensor_size, total_mean,
                  p1=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p2=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p3=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES])
                  ):

    x_train = X[ids, :]
    x_train = rgb2gray(x_train)

    x_tensor = to_tensor(x_train, to_tensor_size)
    x_tensor = x_tensor.unsqueeze(0)
    images = x_tensor.transpose(0, 1)
    images = images.transpose(2, 3)

    #show_gray(images[0])

    replace = True
    if random.uniform(0, 1) > 0.5:
        replace = False

    preds_1, preds_2, preds_3, image_6, total_loss = encode_4_patches(images, colons, optimizers, train, p1, p2, p3)

    m_preds = (preds_1 + preds_2 + preds_3) / 3
    total_mean = 0.99 * total_mean + 0.01 * m_preds.mean(dim=0).detach()

    return preds_1, preds_2, preds_3, total_loss, total_mean, image_6


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


def measure_acc_augments(X_test, colons, targets, total_mean):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    runs = len(X_test)//BATCH_SIZE_DEFAULT
    avg_loss = 0

    print("total mean", total_mean.data.cpu().numpy())
    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        optimizers = []
        p1, p2, p3, mim, total_mean, _ = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, total_mean)
        p1, p2, p3, mim, total_mean, _ = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, total_mean, p1, p2, p3)

        if j == 0:
            print("a prediction 1: ", p1[0].data.cpu().numpy())
            print("a prediction 2: ", p2[0].data.cpu().numpy())
            print("a prediction 3: ", p3[0].data.cpu().numpy())

            print()

            print("a prediction 1: ", p1[20].data.cpu().numpy())
            print("a prediction 2: ", p2[20].data.cpu().numpy())
            print("a prediction 3: ", p3[20].data.cpu().numpy())


        avg_loss += mim.item()
        for i in range(p1.shape[0]):
            val, index = torch.max(p1[i], 0)
            val, index2 = torch.max(p2[i], 0)
            val, index3 = torch.max(p3[i], 0)

            preds = [index.data.cpu().numpy(), index2.data.cpu().numpy(), index3.data.cpu().numpy()]

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
    clusters = set()
    for element in print_dict.keys():
        length = len(print_dict[element])
        misses = miss_classifications(print_dict[element])
        total_miss += misses

        mfe = most_frequent(print_dict[element])
        clusters.add(mfe)
        print("cluster: ",
              element,
              ", most frequent: ",
              mfe,
              ", miss-classifications: ",
              misses,
              ", miss percentage: ",
              misses / length)

    print()
    print("AUGMENTS avg loss: ", avg_loss / runs,
          " miss: ", total_miss,
          " data: ", runs * BATCH_SIZE_DEFAULT,
          " miss percent: ", total_miss / (runs * BATCH_SIZE_DEFAULT))
    print("Clusters found: " + str(len(clusters)) + " " + str(clusters))
    print()

    return avg_loss/runs, len(clusters)


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

    c = OneNetEntropy(1)
    c = c.cuda()
    colons.append(c)
    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    if NETS == 3:
        c1 = OneNetEntropy(1)
        c1 = c1.cuda()
        colons.append(c1)
        optimizer1 = torch.optim.Adam(c1.parameters(), lr=LEARNING_RATE_DEFAULT)
        optimizers.append(optimizer1)

        c2 = OneNetEntropy(1)
        c2 = c2.cuda()
        colons.append(c2)
        optimizer2 = torch.optim.Adam(c2.parameters(), lr=LEARNING_RATE_DEFAULT)
        optimizers.append(optimizer2)

    # c3 = OneNet(1, INPUT_NET)
    # c3 = c3.cuda()
    # colons.append(c3)
    # optimizer3 = torch.optim.Adam(c3.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer3)

    gen = OneNetVAE(SIZE * SIZE + NUMBER_CLASSES)
    gen = gen.cuda()
    colons.append(gen)
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer_gen)

    max_loss = 1999
    max_loss_iter = 0
    most_clusters = 0
    most_clusters_iter = 0

    print(colons[0])
    total_mean = torch.ones([NUMBER_CLASSES]) * 0.1
    total_mean = total_mean.to('cuda')
    print(total_mean)
    print(total_mean.shape)

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, p3, mim, total_mean, image_6 = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, total_mean)
        p1, p2, p3, mim, total_mean, image_6 = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, total_mean, p1, p2, p3)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            print()
            p1, p2, p3, mim, total_mean, image_6 = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, total_mean)
            print("loss 1: ", mim.item())
            p1, p2, p3, mim, total_mean, image_6 = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, total_mean, p1, p2, p3)
            print("loss 2: ", mim.item())

            save_image(image_6.cpu().detach()[0], iteration, "image_6")

            print("iteration: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best loss: ", max_loss_iter,
                  ": ", max_loss)
            print("description: ", DESCRIPTION)

            print_info(p1, p2, p3, targets, test_ids)
            loss, clusters = measure_acc_augments(X_test, colons, targets, total_mean)

            if clusters >= most_clusters:
                most_clusters = clusters
                most_clusters_iter = iteration

            print("most clusters: " + str(most_clusters) + " at iter: " + str(most_clusters_iter))
            if max_loss > loss:
                max_loss = loss
                max_loss_iter = iteration
                #measure_accuracy(X_test, colons, targets)

                print("models saved iter: " + str(iteration))
                # for i in range(number_colons):
                #     torch.save(colons[i], colons_paths[i])

            # print("test loss " + str(test_loss))
            # print("")


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


def print_info(p1, p2, p3, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}
    for i in range(p1.shape[0]):
        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)
        val, index3 = torch.max(p3[i], 0)
        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + " " + str(
            index3.data.cpu().numpy()) + ", "

        label = targets[test_ids[i]]
        if label == 10:
            label = 0
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