from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import os
import torch

from torch.autograd import Variable
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from transform_utils import scale, rotate, random_erease, vertical_flip
from SocialColon import SocialColon
import random
# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 100
EVAL_FREQ_DEFAULT = 200

FLAGS = None

EPS=sys.float_info.epsilon

def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def encode_4_patches(image, colons, p1, p2, p3, p4):

    replace = True
    if random.uniform(0, 1) > 0.5:
        replace = False

    i_1, i_2, i_3, i_4 = split_image_to_4(image, colons[0], colons[0], replace)

    p1 = p1.to('cuda')
    p2 = p2.to('cuda')
    p3 = p3.to('cuda')
    p4 = p4.to('cuda')

    # ids = np.random.choice(len(colons), size=4, replace=False)

    # pred_1 = colons[ids[0]](i_1, p2, p3, p4)
    # pred_2 = colons[ids[1]](i_2, p1, p3, p4)
    # pred_3 = colons[ids[2]](i_3, p1, p2, p4)
    # pred_4 = colons[ids[3]](i_4, p1, p2, p3)

    pred_1, _ = colons[0](i_1, p2, p3, p4)
    pred_2, _ = colons[0](i_2, p1, p3, p4)
    pred_3, _ = colons[0](i_3, p1, p2, p4)
    pred_4, _ = colons[0](i_4, p1, p2, p3)

    return pred_1, pred_2, pred_3, pred_4, i_4


def forward_block(X, ids, colons, optimizers, train, to_tensor_size, mean,
                     p1=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p2=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p3=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p4=torch.zeros([BATCH_SIZE_DEFAULT, 10])):

    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, to_tensor_size)

    images = x_tensor/255

    pred_1, pred_2, pred_3, pred_4, i_4 = encode_4_patches(images, colons, p1, p2, p3, p4)

    product = pred_1 * pred_2 * pred_3 #* pred_4
    product = product.mean(dim=0)
    log_product = torch.log(product)
    loss = -log_product.mean(dim=0)

    # mean_preds = (pred_1 + pred_2 + pred_3 + pred_4)/4
    #
    # H = - (mean_preds * torch.log(mean_preds)).sum(dim=1).mean(dim=0)
    #
    # batch_mean_preds = mean_preds.mean(dim=0)
    # H_batch = - (batch_mean_preds * torch.log(batch_mean_preds)).sum()
    #
    # loss = H - H_batch

    if train:
        for i in optimizers:
            i.zero_grad()

        loss.backward(retain_graph=True)

        for i in optimizers:
            i.step()

    return pred_1, pred_2, pred_3, pred_4, loss, mean


def split_image_to_4(image, vae_enc, vae_dec, replace):
    # image_1 = image
    # image_2 = rotate(image, 20, BATCH_SIZE_DEFAULT)
    # image_3 = scale(image, BATCH_SIZE_DEFAULT)
    # image_4 = random_erease(image, BATCH_SIZE_DEFAULT)

    augments = {
        0: rotate(image, 20, BATCH_SIZE_DEFAULT),
        1: rotate(image, -20, BATCH_SIZE_DEFAULT),
        2: scale(image, BATCH_SIZE_DEFAULT),
        3: image
    }

    ids = np.random.choice(len(augments), size=4, replace=False)

    image_1 = augments[ids[0]]
    image_2 = augments[ids[1]]
    image_3 = augments[ids[2]]
    image_4 = augments[ids[3]]

    # vae_in = torch.reshape(image, (BATCH_SIZE_DEFAULT, 784))
    #
    # sec_mean, sec_std = vae_enc(vae_in)
    # e = torch.zeros(sec_mean.shape).normal_()
    # sec_z = sec_std * e + sec_mean
    # image_4 = vae_dec(sec_z)
    # image_4 = torch.reshape(image_4, (BATCH_SIZE_DEFAULT, 1, 28, 28))

    image_1 = image_1.to('cuda')
    image_2 = image_2.to('cuda')
    image_3 = image_3.to('cuda')
    image_4 = image_4.to('cuda')

    # image = image.to('cuda')
    # show_mnist(image_1[0], 20, 28)
    # show_mnist(image_1[1], 20, 28)
    # show_mnist(image_1[2], 20, 28)
    # show_mnist(image_1[3], 20, 28)
    #
    # show_mnist(image_2[0], 20, 28)
    # show_mnist(image_2[1], 20, 28)
    # show_mnist(image_2[2], 20, 28)
    # show_mnist(image_2[3], 20, 28)
    #
    # input()
    # print(image_1.shape)
    # print(image_2.shape)
    # print(image_3.shape)
    # print(image_4.shape)
    # input()

    return image_1, image_2, image_3, image_4

def print_params(model):
    for param in model.parameters():
        print(param.data)

def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target[60000:]

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    number_colons = 4

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    colons = []

    optimizers = []
    colons_paths = []

    filepath = 'colons\\colon_' + str(0) + '.model'
    predictor_model = os.path.join(script_directory, filepath)
    colons_paths.append(predictor_model)

    #four_split = 3200
    preds = 30
    two_split = 6430

    #two_split_3_conv = 3840

    # c = Ensemble()
    # c.cuda()

    c = SocialColon(1, two_split)
    c.cuda()
    colons.append(c)

    # c2 = SocialColon(1, two_split)
    # c2.cuda()
    # colons.append(c2)
    #
    # c3 = SocialColon(1, two_split)
    # c3.cuda()
    # colons.append(c3)
    #
    # c4 = SocialColon(1, two_split)
    # c4.cuda()
    # colons.append(c4)

    # ve = VaeEncoder()
    # vd = VaeDecoder()
    # colons.append(ve)
    # colons.append(vd)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    # optimizer2 = torch.optim.Adam(c2.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer2)
    #
    # optimizer3 = torch.optim.Adam(c3.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer3)
    #
    # optimizer4 = torch.optim.Adam(c4.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer4)

    # ve_opt = torch.optim.Adam(ve.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(ve_opt)
    #
    # vd_opt = torch.optim.Adam(vd.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(vd_opt)


    max_loss = 1999
    max_loss_iter = 0
    mean = torch.zeros(10)

    for iteration in range(MAX_STEPS_DEFAULT):
        for c in colons:
            c.train()
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, p3, p4, mim, mean = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, mean)
        #p1, p2, p3, p4, mim, mean = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, mean, p1, p2, p3, p4)
        # p1, p2, p3, p4, mim, i_4 = second_guess(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            for c in colons:
                c.eval()
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, p3, p4, mim, mean = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, mean)

            print("mean: ", mean)
            print("loss 1", mim.item())
            # p1, p2, p3, p4, mim, mean = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, mean, p1, p2, p3, p4)
            # print("mean: ", mean)
            # print("loss 2", mim.item())
            # p1, p2, p3, p4, mim, i_4 = second_guess(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)

            # if iteration > 1200:
            #     print(targets[test_ids[0]])
            #     show_mnist(i_4[0].cpu().detach().numpy(), 28, 28)
            #
            #     print(targets[test_ids[1]])
            #     show_mnist(i_4[1].cpu().detach().numpy(), 28, 28)
            #
            #     print(targets[test_ids[2]])
            #     show_mnist(i_4[2].cpu().detach().numpy(), 28, 28)
            #
            #     print(targets[test_ids[3]])
            #     show_mnist(i_4[3].cpu().detach().numpy(), 28, 28)
            #
            #     print(targets[test_ids[4]])
            #     show_mnist(i_4[4].cpu().detach().numpy(), 28, 28)
            #
            #     print(targets[test_ids[5]])
            #     show_mnist(i_4[5].cpu().detach().numpy(), 28, 28)
            #
            #     print(targets[test_ids[6]])
            #     show_mnist(i_4[6].cpu().detach().numpy(), 28, 28)
            #
            #     print(targets[test_ids[7]])
            #     show_mnist(i_4[7].cpu().detach().numpy(), 28, 28)
            #
            #     print(targets[test_ids[8]])
            #     show_mnist(i_4[8].cpu().detach().numpy(), 28, 28)
            #
            #     print(targets[test_ids[9]])
            #     show_mnist(i_4[9].cpu().detach().numpy(), 28, 28)
            #
            #     print(targets[test_ids[10]])
            #     show_mnist(i_4[10].cpu().detach().numpy(), 28, 28)
            print()
            print()
            print("iteration: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best loss: ", max_loss_iter,
                  ": ", max_loss)

            print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
            for i in range(p1.shape[0]):
                if i == 10:
                    print("")

                val, index = torch.max(p1[i], 0)
                val, index2 = torch.max(p2[i], 0)
                val, index3 = torch.max(p3[i], 0)
                val, index4 = torch.max(p4[i], 0)

                string = str(index.data.cpu().numpy())+" "+ str(index2.data.cpu().numpy()) + " "+\
                         str(index3.data.cpu().numpy())+" "+ str(index4.data.cpu().numpy()) +", "

                print_dict[targets[test_ids[i]]] += string


            for i in print_dict.keys():
                print(i, " : ", print_dict[i])
            print()

            avg_loss = measure_acc_augments(X_test, colons, targets)

            if max_loss > avg_loss:
                max_loss_iter = iteration
                max_loss = avg_loss

                print("models saved iter: " + str(iteration))
                torch.save(colons[0], "MNIST_solver.model")

            print("test loss " + str(avg_loss))
            print("")


def measure_acc_block(X_test, test_ids, colons, BATCH_SIZE_DEFAULT,
                     p1=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p2=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p3=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p4=torch.zeros([BATCH_SIZE_DEFAULT, 10])):

    x_train = X_test[test_ids, :]

    image = to_Tensor(x_train, BATCH_SIZE_DEFAULT)
    image = image.to('cuda')

    p1 = p1.to('cuda')
    p2 = p2.to('cuda')
    p3 = p3.to('cuda')
    p4 = p4.to('cuda')

    ids = np.random.choice(len(colons), size=4, replace=False)

    pred_1, _ = colons[ids[0]](image, p2, p3, p4)
    pred_2, _ = colons[ids[1]](image, p1, p3, p4)
    pred_3, _ = colons[ids[2]](image, p1, p2, p4)
    pred_4, _ = colons[ids[3]](image, p1, p2, p3)

    product = pred_1 * pred_2 * pred_3 * pred_4
    product = product.mean(dim=0)
    log_product = torch.log(product)

    # mean_probs = (pred_1.mean(dim=0) + pred_2.mean(dim=0) + pred_3.mean(dim=0) + pred_4.mean(dim=0)) / 4
    #
    # # momentum_mean_prob = betta * momentum_mean_prob.detach() + (1 - betta) * mean_probs
    #
    # if not train:
    #     print("mean probs", mean_probs)
    #     print("product", product)
    #     print("poduct/mean", product / mean_probs)
    #     print("prod - mean", torch.log(product) - torch.log(mean_probs))
    #
    # log_product = torch.log(product) - torch.log(mean_probs)

    loss = - log_product.mean(dim=0)

    return pred_1, pred_2, pred_3, pred_4, loss


def measure_acc_augments(X_test, colons, targets):
    print_dict = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}
    runs = 10000//BATCH_SIZE_DEFAULT
    avg_loss = 0

    mean = torch.zeros(10)
    mean = mean / 10
    #mean = mean.to('cuda')

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        optimizers = []
        p1, p2, p3, p4, mim, mean = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, mean)
        #p1, p2, p3, p4, mim, mean = forward_block(X_test, test_ids, colons,  optimizers, False, BATCH_SIZE_DEFAULT, mean, p1, p2, p3, p4)
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
            print_dict[targets[test_ids[i]]].append(verdict)

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
    print("avg loss: ", avg_loss / runs)
    print("AUGMENTS miss: ", total_miss)
    print("AUGMENTS datapoints: ", runs * BATCH_SIZE_DEFAULT)
    print("AUGMENTS miss percentage: ", total_miss / (runs * BATCH_SIZE_DEFAULT))
    print()

    return avg_loss / runs

# def measure_accuracy(X_test, colons, targets):
#     print_dict = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}
#     runs = 10000//BATCH_SIZE_DEFAULT
#     avg_loss = 0
#     for j in range(runs):
#         test_ids = range(j*BATCH_SIZE_DEFAULT, (j+1)*BATCH_SIZE_DEFAULT)
#
#         p1, p2, p3, p4, mim = measure_acc_block(X_test, test_ids, colons, BATCH_SIZE_DEFAULT)
#         p1, p2, p3, p4, mim = measure_acc_block(X_test, test_ids, colons, BATCH_SIZE_DEFAULT, p1, p2, p3, p4)
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
#             print_dict[targets[test_ids[i]]].append(verdict)
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


def to_Tensor(X, batch_size=BATCH_SIZE_DEFAULT):
    X = np.reshape(X, (batch_size, 1, 28, 28))
    X = Variable(torch.FloatTensor(X))

    return X


def show_mnist(first_image, width, height):
    pixels = first_image.reshape((width, height))
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