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
from one_net_model import OneNet
from stl_utils import rotate, scale, to_grayscale, random_erease, vertical_flip, horizontal_flip, sobel_filter_y, sobel_filter_x, sobel_total,center_crop
import random
import sys
from one_net_generator import OneNetGen
from IID_loss import IID_loss
import torch.nn as nn
from torchvision import transforms
import torchvision


# Default constants
EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000

BATCH_SIZE_DEFAULT = 68
INPUT_NET = 4608
SIZE = 32
NETS = 1
DESCRIPTION = "Augments: 3 augments then 3 sobels x,y,total. Nets: "+str(NETS) +" net. Loss: total_loss = paired_losses - mean_probs_losses. Image size: " + str(SIZE)

EVAL_FREQ_DEFAULT = 200
NUMBER_CLASSES = 11
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def encode_4_patches(image, colons, replace,
                     p1=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p2=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p3=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p4=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p5=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p6=torch.zeros([BATCH_SIZE_DEFAULT, 10])
                     ):
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
                3: scale(original_image, SIZE-6, 3, BATCH_SIZE_DEFAULT),
                4: rotate(original_image, 20, BATCH_SIZE_DEFAULT),
                5: rotate(original_image, -20, BATCH_SIZE_DEFAULT),
                6: center_crop(image, SIZE, BATCH_SIZE_DEFAULT),
                7: sobel_filter_x(original_image, BATCH_SIZE_DEFAULT),
                8: sobel_filter_y(original_image, BATCH_SIZE_DEFAULT),
                9: sobel_total(original_image, BATCH_SIZE_DEFAULT)
                }

    ids = np.random.choice(len(augments), size=6, replace=False)
    #
    # sobels = {0: sobel_filter_x(augments[ids[0]], BATCH_SIZE_DEFAULT),
    #           1: sobel_total(augments[ids[1]], BATCH_SIZE_DEFAULT),
    #           2: sobel_filter_y(augments[ids[2]], BATCH_SIZE_DEFAULT),
    #           3: augments[ids[3]],
    #           4: sobel_total(augments[ids[4]], BATCH_SIZE_DEFAULT),
    #           5: augments[ids[5]]
    # }


    # sobel_id = np.random.choice(len(sobels), size=6, replace=False)
    #
    # image_1 = sobels[sobel_id[0]]
    # image_2 = sobels[sobel_id[1]]
    # image_3 = sobels[sobel_id[2]]
    # image_4 = sobels[sobel_id[3]]
    # image_5 = sobels[sobel_id[4]]
    # image_6 = sobels[sobel_id[5]]

    image_1 = augments[ids[0]]
    image_2 = augments[ids[1]]
    image_3 = augments[ids[2]]
    image_4 = augments[ids[3]]
    image_5 = augments[ids[4]]
    image_6 = augments[ids[5]]

    # show_gray(image_1)
    # show_gray(image_2)
    # show_gray(image_3)
    # show_gray(image_4)
    # show_gray(image_5)
    # show_gray(image_6)

    p1 = p1.cuda()
    p2 = p2.cuda()
    p3 = p3.cuda()
    p4 = p4.cuda()
    p5 = p5.cuda()
    p6 = p6.cuda()

    #net_id = np.random.choice(len(colons), size=4, replace=False)

    nets = [0, 0, 0]
    if NETS == 3:
        nets = [0, 1, 2]

    preds_1 = make_pred(image_1, colons, nets[0], p2, p3, p4, p5, p6)
    preds_2 = make_pred(image_2, colons, nets[1], p1, p3, p4, p5, p6)
    preds_3 = make_pred(image_3, colons, nets[2], p1, p2, p4, p5, p6)
    preds_4 = make_pred(image_4, colons, nets[0], p1, p2, p3, p5, p6)
    preds_5 = make_pred(image_5, colons, nets[1], p1, p2, p3, p4, p6)
    preds_6 = make_pred(image_6, colons, nets[2], p1, p2, p3, p4, p5)

    return preds_1, preds_2, preds_3, preds_4, preds_5, preds_6


def make_pred(image, colons, net, p1, p2, p3, p4, p5):
    image = image.to('cuda')
    pred = colons[net](image, p1, p2, p3, p4, p5)
    del image
    torch.cuda.empty_cache()

    return pred


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
                  p3=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p4=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p5=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p6=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES])
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

    preds_1, preds_2, preds_3, preds_4, preds_5, preds_6 = encode_4_patches(images, colons, replace, p1, p2, p3, p4, p5, p6)

    m_preds = (preds_1 + preds_2 + preds_3 + preds_4 + preds_5 + preds_6) / 6

    H = pred_entropy(m_preds)

    batch_pred_1 = batch_entropy(preds_1)
    batch_pred_2 = batch_entropy(preds_2)
    batch_pred_3 = batch_entropy(preds_3)
    batch_pred_4 = batch_entropy(preds_4)
    batch_pred_5 = batch_entropy(preds_5)
    batch_pred_6 = batch_entropy(preds_6)

    total_loss = H - batch_pred_1 - batch_pred_2 - batch_pred_3 - batch_pred_4 - batch_pred_5 - batch_pred_6

    # product = preds_1 * preds_2 * preds_3 * preds_4 * preds_5 * preds_6
    # sum_indiv = product.mean(dim=1).mean()
    # mean_0 = product.mean(dim=0).mean()
    #
    # total_loss = mean_0 - sum_indiv

    # product = product.mean(dim=0) #* total_mean.detach()
    # log_product = torch.log(product)
    # total_loss = - log_product.mean(dim=0)

    # pm1 = preds_1.mean(dim=0)
    # pm2 = preds_2.mean(dim=0)
    # pm3 = preds_3.mean(dim=0)
    # pm4 = preds_4.mean(dim=0)
    # pm5 = preds_5.mean(dim=0)
    # pm6 = preds_6.mean(dim=0)
    #
    # mean_pred_1 = torch.log(pm1*pm1*pm1*pm1).sum()
    # mean_pred_2 = torch.log(pm2*pm2*pm2*pm2).sum()
    # mean_pred_3 = torch.log(pm3*pm3*pm3*pm3).sum()
    # mean_pred_4 = torch.log(pm4*pm4*pm4*pm4).sum()
    # mean_pred_5 = torch.log(pm5*pm5*pm5*pm5).sum()
    # mean_pred_6 = torch.log(pm6*pm6*pm6*pm6).sum()
    #
    # mean_pred_1 = preds_1.mean(dim=0)
    # mean_pred_2 = preds_2.mean(dim=0)
    # mean_pred_3 = preds_3.mean(dim=0)
    # mean_pred_4 = preds_4.mean(dim=0)
    # mean_pred_5 = preds_5.mean(dim=0)
    # mean_pred_6 = preds_6.mean(dim=0)
    #
    # sqrt_mean_pred_1 = mean_pred_1 * mean_pred_1
    # sqrt_mean_pred_2 = mean_pred_2 * mean_pred_2
    # sqrt_mean_pred_3 = mean_pred_3 * mean_pred_3
    # sqrt_mean_pred_4 = mean_pred_4 * mean_pred_4
    # sqrt_mean_pred_5 = mean_pred_5 * mean_pred_5
    # sqrt_mean_pred_6 = mean_pred_6 * mean_pred_6

    # sum_mean_pred_1 = sqrt_mean_pred_1.mean()
    # sum_mean_pred_2 = sqrt_mean_pred_2.mean()
    # sum_mean_pred_3 = sqrt_mean_pred_3.mean()
    # sum_mean_pred_4 = sqrt_mean_pred_4.mean()
    # sum_mean_pred_5 = sqrt_mean_pred_5.mean()
    # sum_mean_pred_6 = sqrt_mean_pred_6.mean()

    #total_loss = sum_mean_pred_1 + sum_mean_pred_2 + sum_mean_pred_3 + sum_mean_pred_4 + sum_mean_pred_5 + sum_mean_pred_6 - sum_indiv

    # l12 = my_loss(preds_1, preds_2)
    # l13 = my_loss(preds_1, preds_3)
    # l14 = my_loss(preds_1, preds_4)
    # l15 = my_loss(preds_1, preds_5)
    # l16 = my_loss(preds_1, preds_6)
    #
    # l23 = my_loss(preds_2, preds_3)
    # l24 = my_loss(preds_2, preds_4)
    # l25 = my_loss(preds_2, preds_5)
    # l26 = my_loss(preds_2, preds_6)
    #
    # l34 = my_loss(preds_3, preds_4)
    # l35 = my_loss(preds_3, preds_5)
    # l36 = my_loss(preds_3, preds_6)
    #
    # l45 = my_loss(preds_4, preds_5)
    # l46 = my_loss(preds_4, preds_6)
    #
    # l56 = my_loss(preds_5, preds_6)

    # l12 = IID_loss(preds_1, preds_2)
    # l13 = IID_loss(preds_1, preds_3)
    # l14 = IID_loss(preds_1, preds_4)
    # l15 = IID_loss(preds_1, preds_5)
    # l16 = IID_loss(preds_1, preds_6)
    #
    # l23 = IID_loss(preds_2, preds_3)
    # l24 = IID_loss(preds_2, preds_4)
    # l25 = IID_loss(preds_2, preds_5)
    # l26 = IID_loss(preds_2, preds_6)
    #
    # l34 = IID_loss(preds_3, preds_4)
    # l35 = IID_loss(preds_3, preds_5)
    # l36 = IID_loss(preds_3, preds_6)
    #
    # l45 = IID_loss(preds_4, preds_5)
    # l46 = IID_loss(preds_4, preds_6)
    #
    # l56 = IID_loss(preds_5, preds_6)

    # paired_losses = l12 + l13 + l14 + l15 + l16 + l23 + l24 + l25 + l26 + l34 + l35 + l36 + l45 + l46 + l56
    # mean_probs_losses = mean_pred_1 + mean_pred_2 + mean_pred_3 + mean_pred_4 + mean_pred_5 + mean_pred_6
    # # #H = pred_entropy(m_preds)
    # #
    # total_loss = paired_losses - mean_probs_losses
    # product = m_preds * m_preds
    # product_mean = product.mean(dim=0)
    # log_product_mean = torch.log(product_mean)
    # batch_diversion = - log_product_mean.mean(dim=0)
    #
    # total_loss = batch_diversion

    # H = pred_entropy(m_preds)
    #
    # sum1 = distance_loss(preds_1)
    # sum2 = distance_loss(preds_2)
    # sum3 = distance_loss(preds_3)
    #
    # total_loss = H + sum1 + sum2 + sum3

    # coeff = 3
    # total_loss = entropies_loss(m_preds, coeff)

    total_mean = 0.99 * total_mean + 0.01 * m_preds.mean(dim=0).detach()


    if train:
        torch.autograd.set_detect_anomaly(True)

        for idx, i in enumerate(optimizers):
            i.zero_grad()

        # l1.backward(retain_graph=True)
        # l2.backward(retain_graph=True)
        # l3.backward(retain_graph=True)
        # l4.backward(retain_graph=True)
        total_loss.backward(retain_graph=True)

        for idx, i in enumerate(optimizers):
            i.step()

    return preds_1, preds_2, preds_3, preds_4, preds_5, preds_6, total_loss, total_mean


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


def measure_acc_augments(X_test, colons, targets, total_mean):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    runs = len(X_test)//BATCH_SIZE_DEFAULT
    avg_loss = 0

    print("total mean", total_mean.data.cpu().numpy())
    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        optimizers = []
        p1, p2, p3, p4, p5, p6, mim, total_mean = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, total_mean)
        #p1, p2, p3, p4, p5, p6, mim, total_mean = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, total_mean, p1, p2, p3, p4, p5, p6)

        if j == 0:
            print("a prediction 1: ", p1[0].data.cpu().numpy())
            print("a prediction 2: ", p2[0].data.cpu().numpy())
            print("a prediction 3: ", p3[0].data.cpu().numpy())
            print("a prediction 4: ", p4[0].data.cpu().numpy())
            print("a prediction 5: ", p5[0].data.cpu().numpy())
            print("a prediction 6: ", p6[0].data.cpu().numpy())

            print()

            print("a prediction 1: ", p1[20].data.cpu().numpy())
            print("a prediction 2: ", p2[20].data.cpu().numpy())
            print("a prediction 3: ", p3[20].data.cpu().numpy())
            print("a prediction 4: ", p4[20].data.cpu().numpy())
            print("a prediction 5: ", p5[20].data.cpu().numpy())
            print("a prediction 6: ", p6[20].data.cpu().numpy())

        avg_loss += mim.item()
        for i in range(p1.shape[0]):
            val, index = torch.max(p1[i], 0)
            val, index2 = torch.max(p2[i], 0)
            val, index3 = torch.max(p3[i], 0)
            val, index4 = torch.max(p4[i], 0)
            val, index5 = torch.max(p5[i], 0)
            val, index6 = torch.max(p6[i], 0)

            preds = [index.data.cpu().numpy(), index2.data.cpu().numpy(), index3.data.cpu().numpy(),
                     index4.data.cpu().numpy(), index5.data.cpu().numpy(), index6.data.cpu().numpy()]

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

    c = OneNet(1, INPUT_NET)
    c = c.cuda()
    colons.append(c)
    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    if NETS == 3:
        c1 = OneNet(1, INPUT_NET)
        c1 = c1.cuda()
        colons.append(c1)
        optimizer1 = torch.optim.Adam(c1.parameters(), lr=LEARNING_RATE_DEFAULT)
        optimizers.append(optimizer1)

        c2 = OneNet(1, INPUT_NET)
        c2 = c2.cuda()
        colons.append(c2)
        optimizer2 = torch.optim.Adam(c2.parameters(), lr=LEARNING_RATE_DEFAULT)
        optimizers.append(optimizer2)

    # c3 = OneNet(1, INPUT_NET)
    # c3 = c3.cuda()
    # colons.append(c3)
    # optimizer3 = torch.optim.Adam(c3.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer3)

    gen = OneNetGen(SIZE*SIZE)
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
        p1, p2, p3, p4, p5, p6, mim, total_mean = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, total_mean)
        #p1, p2, p3, p4, p5, p6, mim, total_mean = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, total_mean, p1, p2, p3, p4, p5, p6)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            print()
            p1, p2, p3,  p4, p5, p6, mim, total_mean = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, total_mean,)
            print("loss 1: ", mim.item())
            #p1, p2, p3,  p4, p5, p6,  mim, total_mean = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, total_mean, p1, p2, p3, p4, p5, p6)
            print("loss 2: ", mim.item())


            print("iteration: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best loss: ", max_loss_iter,
                  ": ", max_loss)
            print("description: ", DESCRIPTION)

            print_info(p1, p2, p3, p4, p5, p6, targets, test_ids)
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


def print_info(p1, p2, p3, p4, p5, p6, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}
    for i in range(p1.shape[0]):
        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)
        val, index3 = torch.max(p3[i], 0)
        val, index4 = torch.max(p4[i], 0)
        val, index5 = torch.max(p5[i], 0)
        val, index6 = torch.max(p6[i], 0)

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + " " + str(
            index3.data.cpu().numpy()) + " " + str(index4.data.cpu().numpy()) + " " + str(index5.data.cpu().numpy()) + " " + str(index6.data.cpu().numpy()) + ", "

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