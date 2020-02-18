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
from stl_utils import rotate, scale, to_grayscale, random_erease, vertical_flip, horizontal_flip
import random
import sys
import torch.nn as nn
from torchvision import transforms
import torchvision


# Default constants
EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 48
EVAL_FREQ_DEFAULT = 100
NUMBER_CLASSES = 1
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
                     p4=torch.zeros([BATCH_SIZE_DEFAULT, 10])):
    image /= 255

    size = 50
    pad = (96-size)//2
    original_image = scale(image, size, pad, BATCH_SIZE_DEFAULT)
    show_gray(original_image)

    original_image = original_image[:, :, pad:96-pad, pad:96-pad]
    show_gray(original_image)

    # augments = {0: rotate(original_image, 15, BATCH_SIZE_DEFAULT),
    #             1: rotate(original_image, -15, BATCH_SIZE_DEFAULT),
    #             2: scale(original_image, 30, 5, BATCH_SIZE_DEFAULT),
    #             3: vertical_flip(original_image, BATCH_SIZE_DEFAULT),
    #             4: original_image}

    augments = {0: scale(original_image, size-10, 5, BATCH_SIZE_DEFAULT),
                1: horizontal_flip(original_image, BATCH_SIZE_DEFAULT),
                2: original_image,
                3: horizontal_flip(scale(original_image, size-10, 5, BATCH_SIZE_DEFAULT), BATCH_SIZE_DEFAULT),
                4: rotate(original_image, 20, BATCH_SIZE_DEFAULT),
                5: rotate(original_image, -20, BATCH_SIZE_DEFAULT),
                6: vertical_flip(original_image, BATCH_SIZE_DEFAULT)}

    ids = np.random.choice(len(augments), size=4, replace=False)

    image_1 = augments[ids[0]]
    image_2 = augments[ids[1]]
    image_3 = augments[ids[2]]
    image_4 = augments[ids[3]]

    show_gray(image_1)
    show_gray(image_2)
    show_gray(image_3)
    show_gray(image_4)

    p1 = p1.cuda()
    p2 = p2.cuda()
    p3 = p3.cuda()
    p4 = p4.cuda()

    #net_id = np.random.choice(len(colons), size=4, replace=False)

    image_1 = image_1.to('cuda')
    image_2 = image_2.to('cuda')
    image_3 = image_3.to('cuda')
    image_4 = image_4.to('cuda')

    preds_1 = colons[0](image_1, p2, p3, p4)
    preds_2 = colons[0](image_2, p1, p3, p4)
    preds_3 = colons[0](image_3, p1, p2, p4)
    preds_4 = colons[0](image_4, p1, p2, p3)

    # image_1 = image_1.to('cuda')
    # preds_1 = colons[0](image_1, p2, p3, p4)
    # del image_1
    # torch.cuda.empty_cache()
    #
    # image_2 = image_2.to('cuda')
    # preds_2 = colons[0](image_2, p1, p3, p4)
    # del image_2
    # torch.cuda.empty_cache()
    #
    # image_3 = image_3.to('cuda')
    # preds_3 = colons[0](image_3, p1, p2, p4)
    # del image_3
    # torch.cuda.empty_cache()
    #
    # image_4 = image_4.to('cuda')
    # preds_4 = colons[0](image_4, p1, p2, p3)
    # del image_4
    # torch.cuda.empty_cache()

    # image_1 = image_1.to('cuda')
    # preds_1 = colons[net_id[0]](image_1, p2, p3, p4)
    # del image_1
    # torch.cuda.empty_cache()
    #
    # image_2 = image_2.to('cuda')
    # preds_2 = colons[net_id[1]](image_2, p1, p3, p4)
    # del image_2
    # torch.cuda.empty_cache()
    #
    # image_3 = image_3.to('cuda')
    # preds_3 = colons[net_id[2]](image_3, p1, p2, p4)
    # del image_3
    # torch.cuda.empty_cache()
    #
    # image_4 = image_4.to('cuda')
    # preds_4 = colons[net_id[3]](image_4, p1, p2, p3)
    # del image_4
    # torch.cuda.empty_cache()

    return preds_1, preds_2, preds_3, preds_4


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


def forward_block(X, ids, colons, optimizers, train, to_tensor_size,
                p1 = torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p2 = torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                          p3 = torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                               p4 = torch.zeros([BATCH_SIZE_DEFAULT, 10])

                            ):

    x_train = X[ids, :]
    x_train = rgb2gray(x_train)

    x_tensor = to_tensor(x_train, to_tensor_size)
    x_tensor = x_tensor.unsqueeze(0)
    images = x_tensor.transpose(0, 1)
    images = images.transpose(2, 3)

    show_gray(images[0])


    x = images
    print("x shape", x.shape)

    ##### x derivative ####
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])

    a = a.view((1, 1, 3, 3))

    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(a)

    G_x = conv1(Variable(x)).data.view(BATCH_SIZE_DEFAULT, 1, 96, 96)

    ##### y derivative ####

    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])

    b = b.view((1, 1, 3, 3))

    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(b)
    G_y = conv2(Variable(x)).data.view(BATCH_SIZE_DEFAULT, 1, 96, 96)

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

    images = G_x
    show_gray(images[0])

    # print("G shape", G.shape)
    show_gray(G_y[0])
    # show_gray(G_x[4])
    # show_gray(G[5])
    #
    # input()
    #######

    #images = x_tensor/255.0

    replace = True
    if random.uniform(0, 1) > 0.5:
        replace = False

    preds_1, preds_2, preds_3, preds_4 = encode_4_patches(images, colons, replace, p1, p2, p3, p4)

    #mean_preds = preds_1 * preds_2 * preds_3 * preds_4
    m_preds = (preds_1 + preds_2 + preds_3 + preds_4) / 4

    coeff = 1
    total_loss = entropies_loss(m_preds, coeff)

    l1 = entropies_loss(preds_1, coeff)
    l2 = entropies_loss(preds_2, coeff)
    l3 = entropies_loss(preds_3, coeff)
    l4 = entropies_loss(preds_4, coeff)

    #TODO
    total_loss = l1 + l2 + l3 + l4 + total_loss



    # product = mean_preds * mean_preds
    #
    # product = product.mean(dim=0)
    # log_product = torch.log(product)
    # total_loss = - log_product.mean(dim=0)

    if train:
        torch.autograd.set_detect_anomaly(True)

        for idx, i in enumerate(optimizers):
            i.zero_grad()

        total_loss.backward(retain_graph=True)

        for idx, i in enumerate(optimizers):
            i.step()

    return preds_1, preds_2, preds_3, preds_4, total_loss


def entropies_loss(pred, coeff):
    return pred_entropy(pred) - coeff * batch_entropy(pred)


def pred_entropy(prediction):
    pred = prediction + EPS
    H = - (pred * torch.log(pred)).sum(dim=1).mean(dim=0)
    return H


def batch_entropy(pred):
    batch_mean_preds = pred.mean(dim=0)
    H_batch = - (batch_mean_preds * torch.log(batch_mean_preds)).sum()

    return H_batch


# def measure_acc_block(X_test, test_ids, colons, BATCH_SIZE_DEFAULT):
#
#     x_train = X_test[test_ids, :]
#
#     image = to_tensor(x_train, BATCH_SIZE_DEFAULT)
#     image = image.to('cuda')
#
#     # p1 = p1.to('cuda')
#     # p2 = p2.to('cuda')
#     # p3 = p3.to('cuda')
#     # p4 = p4.to('cuda')
#
#     ids = np.random.choice(len(colons), size=4, replace=False)
#
#     pred_1 = colons[ids[0]](image)
#     pred_2 = colons[ids[1]](image)
#     pred_3 = colons[ids[2]](image)
#     pred_4 = colons[ids[3]](image)
#
#     product = pred_1 * pred_2 * pred_3 * pred_4
#     product = product.mean(dim=0)
#     log_product = torch.log(product)
#
#     # mean_probs = (pred_1.mean(dim=0) + pred_2.mean(dim=0) + pred_3.mean(dim=0) + pred_4.mean(dim=0)) / 4
#     #
#     # # momentum_mean_prob = betta * momentum_mean_prob.detach() + (1 - betta) * mean_probs
#     #
#     # if not train:
#     #     print("mean probs", mean_probs)
#     #     print("product", product)
#     #     print("poduct/mean", product / mean_probs)
#     #     print("prod - mean", torch.log(product) - torch.log(mean_probs))
#     #
#     # log_product = torch.log(product) - torch.log(mean_probs)
#
#     loss = - log_product.mean(dim=0)
#
#     return pred_1, pred_2, pred_3, pred_4, loss


def measure_acc_augments(X_test, colons, targets):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    runs = len(X_test)//BATCH_SIZE_DEFAULT
    avg_loss = 0

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        optimizers = []
        p1, p2, p3, p4, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)
        #p1, p2, p3, p4, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)

        avg_loss += mim.item()
        for i in range(p1.shape[0]):
            val, index = torch.max(p1[i], 0)
            val, index2 = torch.max(p2[i], 0)
            val, index3 = torch.max(p3[i], 0)
            val, index4 = torch.max(p4[i], 0)

            preds = [index.data.cpu().numpy(), index2.data.cpu().numpy(), index3.data.cpu().numpy(),
                     index4.data.cpu().numpy()]
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
#
#
# def measure_accuracy(X_test, colons, targets):
#     print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
#     runs = len(X_test)//BATCH_SIZE_DEFAULT
#     avg_loss = 0
#     for j in range(runs):
#         test_ids = range(j*BATCH_SIZE_DEFAULT, (j+1)*BATCH_SIZE_DEFAULT)
#
#         p1, p2, p3, p4, mim = measure_acc_block(X_test, test_ids, colons, BATCH_SIZE_DEFAULT)
#
#         avg_loss += mim.item()
#         for i in range(p1.shape[0]):
#
#             val, index = torch.max(p1[i], 0)
#             val, index2 = torch.max(p2[i], 0)
#             val, index3 = torch.max(p3[i], 0)
#             val, index4 = torch.max(p4[i], 0)
#
#             preds = [index.data.cpu().numpy(), index2.data.cpu().numpy(), index3.data.cpu().numpy(), index4.data.cpu().numpy()]
#             preds = list(preds)
#             preds = [int(x) for x in preds]
#             #print(preds)
#             verdict = most_frequent(preds)
#
#             # print("verdict", verdict)
#             # print("target", targets[test_ids[i]])
#             # input()
#             label = targets[test_ids[i]]
#             if label == 10:
#                 label = 0
#
#             print_dict[label].append(verdict)
#
#     total_miss = 0
#     for element in print_dict.keys():
#         length = len(print_dict[element])
#         misses = miss_classifications(print_dict[element])
#         total_miss += misses
#
#         print("cluster: ",
#               element,
#               ", most frequent: ",
#               most_frequent(print_dict[element]),
#               ", miss-classifications: ",
#               misses,
#               ", miss percentage: ",
#               misses/length)
#
#     print()
#     print("avg loss: ", avg_loss/runs)
#     print("TOTAL miss: ", total_miss)
#     print("TOTAL datapoints: ", runs*BATCH_SIZE_DEFAULT)
#     print("TOTAL miss percentage: ", total_miss/(runs*BATCH_SIZE_DEFAULT))
#     print()


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

    input = 12800
    #input = 1152

    c = OneNet(1, input)
    c = c.cuda()
    colons.append(c)

    # c1 = OneNet(1, input)
    # c1 = c1.cuda()
    # colons.append(c1)
    #
    # c2 = OneNet(1, input)
    # c2 = c2.cuda()
    # colons.append(c2)
    #
    # c3 = OneNet(1, input)
    # c3 = c3.cuda()
    # colons.append(c3)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    # optimizer1 = torch.optim.Adam(c1.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer1)
    #
    # optimizer2 = torch.optim.Adam(c2.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer2)
    #
    # optimizer3 = torch.optim.Adam(c3.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer3)

    max_loss = 1999
    max_loss_iter = 0
    description = "4 augments"
    print(colons[0])

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, p3, p4, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)
        #p1, p2, p3, p4, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            print()
            p1, p2, p3, p4, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)
            print("loss 1: ", mim.item())
            #p1, p2, p3, p4, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)
            #print("loss 2: ", mim.item())


            print("iteration: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best loss: ", max_loss_iter,
                  ": ", max_loss)
            print("description: ", description)


            print_info(p1, p2, p3, p4, targets, test_ids)

            test_loss = mim.item()

            if max_loss > test_loss:
                max_loss = test_loss
                max_loss_iter = iteration
                #measure_accuracy(X_test, colons, targets)

                measure_acc_augments(X_test, colons, targets)


                print("models saved iter: " + str(iteration))
                # for i in range(number_colons):
                #     torch.save(colons[i], colons_paths[i])

            print("test loss " + str(test_loss))
            print("")


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


def print_info(p1, p2, p3, p4, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}
    for i in range(p1.shape[0]):
        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)
        val, index3 = torch.max(p3[i], 0)
        val, index4 = torch.max(p4[i], 0)

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + " " + str(
            index3.data.cpu().numpy())+ " "  + str(index4.data.cpu().numpy()) + ", "

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