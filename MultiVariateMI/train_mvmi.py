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
from mlp import ColonMLP
import sys
from colon_mvmi import ColonMVMI

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 100
EVAL_FREQ_DEFAULT = 200
EPS = sys.float_info.epsilon
FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def encode_4_patches(image, colons):
    i_1, i_2, i_3, i_4 = split_image_to_4(image)

    pred_1 = colons[0](i_1)
    pred_2 = colons[0](i_2)
    pred_3 = colons[0](i_3)
    pred_4 = colons[0](i_4)

    return pred_1, pred_2, pred_3, pred_4


def encode_3_patches(image, colons):
    # i_1, i_2, i_3 = split_image_to_3(image)
    #
    # flat_1 = torch.flatten(i_1, 1)
    # flat_2 = torch.flatten(i_2, 1)
    # flat_3 = torch.flatten(i_3, 1)
    #
    # pred_1 = colons[0](i_1)
    # pred_2 = colons[0](i_2)
    # pred_3 = colons[0](i_3)

    i_1, i_2, i_3, i_4 = split_image_to_4(image)

    # TODO fix
    pred_1 = colons[0](i_1)
    pred_2 = colons[1](i_2)
    pred_3 = colons[2](i_3)
    pred_4 = colons[3](i_4)

    # image = image.to('cuda')
    # pred_1, pred_2, pred_3 = colons[0](image)

    # print("pred_1 ", pred_1.shape)
    # print("pred_2 ", pred_2.shape)
    # print("pred_3 ", pred_3.shape)
    # print(pred_1[0])
    # input()

    return pred_1, pred_2, pred_3, pred_4


def squash(input_tensor, dim=1):
    squared_norm = (input_tensor ** 2).sum(dim, keepdim=True)
    output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))

    return output_tensor


def forward_block(X, ids, colons, optimizers, train, to_tensor_size):
    x_train = X[ids, :]

    x_tensor = to_tensor(x_train, to_tensor_size)

    images = x_tensor/255

    # pred_1, pred_2, pred_3 = encode_3_patches(images, colons)
    # loss = three_variate_IID_loss(pred_1, pred_2, pred_3)

    # pred_1, pred_2, pred_3, pred_4 = encode_4_patches(images, colons)


    pred_1, pred_2, pred_3, pred_4 = encode_4_patches(images, colons)
    #
    # s1 = squash(pred_1)
    # s2 = squash(pred_2)
    # s3 = squash(pred_3)
    # s4 = squash(pred_4)

    # s1 = pred_1
    # s2 = pred_2
    # s3 = pred_3
    # s4 = pred_4

    # print(s1.shape)
    # print(s2.shape)
    #

    # z2 = torch.matmul(s3, s4.transpose(0, 1))
    #
    # print(z.shape)
    # input()

    # prod = s1 * s2 * s3 * s4
    # prod = prod.mean(dim=0)
    # prod[(prod < EPS).data] = EPS
    #
    # log_prod = torch.log(prod)
    # log_prod[(log_prod < EPS).data] = EPS
    # #p = torch.sqrt((prod ** 2).sum(1, keepdim=True))
    # #print(log_prod)
    # p = - log_prod.mean(dim=0)


    # if train:
    #     optimizers[0].zero_grad()
    #     p.backward(retain_graph=True)
    #     optimizers[0].step()

    # dim = 1
    # pred_1 = torch.sqrt((pred_1 ** 2).sum(dim, keepdim=True))

    prediction_1 = colons[1](pred_1)
    prediction_2 = colons[1](pred_2)
    prediction_3 = colons[1](pred_3)
    prediction_4 = colons[1](pred_4)

    predictions = prediction_1 * prediction_2 * prediction_3 * prediction_4
    product = predictions.mean(dim=0)
    log_product = torch.log(product)
    loss = -log_product.mean(dim=0)

    if train:
        optimizers[0].zero_grad()
        optimizers[1].zero_grad()

        loss.backward(retain_graph=True)

        optimizers[0].step()
        optimizers[1].step()

    #loss = four_variate_IID_loss(pred_1, pred_2, pred_3, pred_4)

    # loss_1 = IID_loss(pred_1, pred_2)
    # loss_2 = IID_loss(pred_3, pred_4)

    # loss_3 = IID_loss(pred_1, pred_3)
    # loss_4 = IID_loss(pred_1, pred_4)
    #
    # loss_5 = IID_loss(pred_2, pred_3)
    # loss_6 = IID_loss(pred_2, pred_4)

    # loss_a = loss_1  #+ loss_3 + loss_4 + loss_5 + loss_6
    # loss_b = loss_2  #+ loss_3 + loss_4 + loss_5 + loss_6

    # if train:
    #     for i in optimizers:
    #         i.zero_grad()
    #
    #     loss.backward(retain_graph=True)
    #
    #     for i in optimizers:
    #         i.step()

    return prediction_1, prediction_2, prediction_3, prediction_4, loss, loss


def split_image_to_3(images):
    image_shape = images.shape

    image_a, image_b = torch.split(images, image_shape[2] // 2, dim=3)
    image_3, image_4 = torch.split(images, image_shape[2] // 2, dim=2)

    image_a = image_a.to('cuda')
    image_b = image_b.to('cuda')
    image_4 = image_4.to('cuda')

    # print(images.shape)
    # print("image a batch: ", image_a.shape)
    # print("image b batch: ", image_b.shape)
    # print("image 3 batch: ", image_3.shape)
    # print("image 4 batch: ", image_4.shape)

    return image_a, image_b, image_4


def split_image_to_4(image):
    split_at_pixel = 20
    width = image.shape[2]
    height = image.shape[3]
    #
    image_1 = image[:, :, 0: split_at_pixel, :]
    image_2 = image[:, :, width - split_at_pixel:, :]
    image_3 = image[:, :, :, 0: split_at_pixel]
    image_4 = image[:, :, :, height - split_at_pixel:]

    # # image_1, _ = torch.split(image, split_at_pixel, dim=3)
    # # image_3, _ = torch.split(image, split_at_pixel, dim=2)
    #
    image_1 = image_1.to('cuda')
    image_2 = image_2.to('cuda')
    image_3 = image_3.to('cuda')
    image_4 = image_4.to('cuda')

    #image = image.to('cuda')
    # show_mnist(image_1[0], split_at_pixel, 28)
    # show_mnist(image_1[1], split_at_pixel, 28)
    # show_mnist(image_1[2], split_at_pixel, 28)
    # show_mnist(image_1[3], split_at_pixel, 28)
    #
    # show_mnist(image_2[0], split_at_pixel, 28)
    # show_mnist(image_2[1], split_at_pixel, 28)
    # show_mnist(image_2[2], split_at_pixel, 28)
    # show_mnist(image_2[3], split_at_pixel, 28)
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

    input = 5120
    input = 3840

    # c = Ensemble()
    # c.cuda()

    c = ColonMVMI(1, input)
    c.cuda()
    colons.append(c)

    c2 = ColonMLP(1, 71680)
    c2.cuda()
    colons.append(c2)
    #
    # c3 = ColonMVMI(1, input)
    # c3.cuda()
    # colons.append(c3)
    #
    # c4 = ColonMVMI(1, input)
    # c4.cuda()
    # colons.append(c4)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    optimizer2 = torch.optim.Adam(c2.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer2)
    #
    # optimizer3 = torch.optim.Adam(c3.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer3)
    #
    # optimizer4 = torch.optim.Adam(c4.parameters(), lr=LEARNING_RATE_DEFAULT)
    # optimizers.append(optimizer4)

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        prediction_1, prediction_2, prediction_3, prediction_4, p, loss = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            prediction_1, prediction_2, prediction_3, prediction_4, p, loss = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)
            print()
            print("iteration: ", iteration)
            print("p loss: ", p.item())

            print_info(prediction_1, prediction_2, prediction_3, prediction_4, targets, test_ids)

            test_loss = loss.item()

            if max_loss > test_loss:
                max_loss = test_loss
                print("models saved iter: " + str(iteration))
                # for i in range(number_colons):
                #     torch.save(colons[i], colons_paths[i])

            print("test loss " + str(test_loss))
            print("")


def to_tensor(X, batch_size=BATCH_SIZE_DEFAULT):
    X = np.reshape(X, (batch_size, 1, 28, 28))
    X = Variable(torch.FloatTensor(X))

    return X


def show_mnist(first_image, w ,h):
    pixels = first_image.reshape((w, h))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def print_info(p1, p2, p3, p4, targets, test_ids):
    print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
    for i in range(BATCH_SIZE_DEFAULT):
        if i == 10:
            print("")

        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)
        val, index3 = torch.max(p3[i], 0)
        val, index4 = torch.max(p4[i], 0)

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + " " + \
                 str(index3.data.cpu().numpy()) + " " + str(index4.data.cpu().numpy()) + ", "

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