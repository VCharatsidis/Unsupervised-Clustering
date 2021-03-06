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

from transform_utils import scale, rotate, random_erease, vertical_flip

from ten_sigmoid import TenSigmoid

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 120
EVAL_FREQ_DEFAULT = 200

FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def encode_4_patches(image, colons):

    i_1, i_2, i_3, i_4 = split_image_to_4(image)

    pred_1 = colons[0](i_1)
    pred_2 = colons[0](i_2)
    pred_3 = colons[0](i_3)
    pred_4 = colons[0](i_4)

    return pred_1, pred_2, pred_3, pred_4


def forward_block(X, ids, colons, optimizers, train, to_tensor_size):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, to_tensor_size)

    images = x_tensor/255

    pred_1, pred_2, pred_3, pred_4 = encode_4_patches(images, colons)

    total_preds = pred_1 * pred_2 * pred_3 * pred_4
    total_preds = total_preds.to('cpu')

    products = []
    classes = 10
    for prod in range(classes):
        product = total_preds[:, prod].clone()

        for idx in range(classes):
            if idx != prod:
                product *= torch.ones(total_preds[:, idx].shape) - total_preds[:, idx].clone()
            else:
                product *= total_preds[:, idx].clone()

        products.append(product)

    loss = torch.zeros([1])

    for p in products:
        mean = p.mean(dim=0)
        log_p = -torch.log(mean)
        loss += log_p

    loss /= 10

    if train:
        for i in optimizers:
            i.zero_grad()

        loss.backward(retain_graph=True)

        for i in optimizers:
            i.step()

    return pred_1, pred_2, pred_3, pred_4, loss


def split_image_to_4(image):
    # image_1 = image
    # image_2 = rotate(image, 20, BATCH_SIZE_DEFAULT)
    # image_3 = scale(image, BATCH_SIZE_DEFAULT)
    # image_4 = random_erease(image, BATCH_SIZE_DEFAULT)

    augments = {
        0: rotate(image, 20, BATCH_SIZE_DEFAULT),
        1: rotate(image, -20, BATCH_SIZE_DEFAULT),
        2: scale(image, BATCH_SIZE_DEFAULT),
        3: vertical_flip(image, BATCH_SIZE_DEFAULT),
        4: scale(image, BATCH_SIZE_DEFAULT),
        5: random_erease(image, BATCH_SIZE_DEFAULT),
        6: image
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


    two_split = 6400

    c = TenSigmoid(1, two_split)
    c.cuda()
    colons.append(c)

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

    max_loss = 1000000
    max_loss_iter = 0

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, p3, p4, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, p3, p4, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)

            print()
            print("iteration: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best loss: ", max_loss_iter,
                  ": ", max_loss)

            print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
            for i in range(p1.shape[0]):

                val, index = torch.max(p1[i], 0)
                val, index2 = torch.max(p2[i], 0)
                val, index3 = torch.max(p3[i], 0)
                val, index4 = torch.max(p4[i], 0)

                string = str(index.data.cpu().numpy())+" "+ str(index2.data.cpu().numpy()) + " "+\
                         str(index3.data.cpu().numpy())+" "+ str(index4.data.cpu().numpy()) +", "

                print_dict[targets[test_ids[i]]] += string


            for i in print_dict.keys():
                print(i, " : ", print_dict[i])

            test_loss = mim.item()

            if max_loss > test_loss:
                max_loss_iter = iteration
                max_loss = test_loss
                print("models saved iter: " + str(iteration))
                # for i in range(number_colons):
                #     torch.save(colons[i], colons_paths[i])

            print("test loss " + str(test_loss))
            print("")


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