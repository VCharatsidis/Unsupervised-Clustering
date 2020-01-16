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
from multi_variate_mi import three_variate_IID_loss
from ensemble import Ensemble

from colon import Colon

# Default constants
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 800
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


def encode_4_patches(image, colons):
    i_1, i_2, i_3, i_4 = split_image_to_4(image)
    print(i_1.shape)
    flat_1 = torch.flatten(i_1)
    flat_2 = torch.flatten(i_2)
    flat_3 = torch.flatten(i_3)
    flat_4 = torch.flatten(i_4)
    print(flat_1.shape)

    pred_1 = colons[0](flat_1)
    pred_2 = colons[1](flat_2)
    pred_3 = colons[2](flat_3)
    pred_4 = colons[3](flat_4)

    return pred_1, pred_2, pred_3, pred_4


def encode_3_patches(image, colons):
    # i_1, i_2, i_3 = split_image_to_3(image)

    # flat_1 = torch.flatten(i_1, 1)
    # flat_2 = torch.flatten(i_2, 1)
    # flat_3 = torch.flatten(i_3, 1)

    # print("flat 1 ", flat_1.shape)
    # print("flat 2 ", flat_2.shape)
    # print("flat 3 ", flat_3.shape)

    # pred_1 = colons[0](i_1)
    # pred_2 = colons[0](i_2)
    # pred_3 = colons[0](i_3)

    image = image.to('cuda')
    pred_1, pred_2, pred_3 = colons[0](image)

    # print("pred_1 ", pred_1.shape)
    # print("pred_2 ", pred_2.shape)
    # print("pred_3 ", pred_3.shape)
    # print(pred_1[0])
    # input()

    return pred_1, pred_2, pred_3


def forward_block(X, ids, colons, optimizers, train, to_tensor_size):
    x_train = X[ids, :]

    x_tensor = to_Tensor(x_train, to_tensor_size)

    images = x_tensor/255

    pred_1, pred_2, pred_3 = encode_3_patches(images, colons)

    loss = three_variate_IID_loss(pred_1, pred_2, pred_3)

    if train:
        optimizers[0].zero_grad()
        loss.backward(retain_graph=True)
        optimizers[0].step()

    return pred_1, pred_2, pred_3, loss


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
    image_shape = image.shape

    image_a, image_b = torch.split(image, image_shape[2]//2, dim=2)

    image_1, image_2 = torch.split(image_a, image_shape[2]//2, dim=3)
    image_3, image_4 = torch.split(image_b, image_shape[2]//2, dim=3)
    # print(image_1.shape)
    # print(image_2.shape)
    # print(image_3.shape)
    # print(image_4.shape)
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

    #input = 3840
    input = 5120

    c = Ensemble()
    c.cuda()

    # c = Colon(1, input)
    # c.cuda()
    colons.append(c)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, p3, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, p3, mim = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)
            print()
            print("iteration: ", iteration)

            for i in range(25):
                if i == 10:
                    print("")

                val, index = torch.max(p1[i], 0)
                val, index2 = torch.max(p2[i], 0)
                val, index3 = torch.max(p3[i], 0)

                print(targets[test_ids[i]], ": ", index.data.cpu().numpy(), " ", index2.data.cpu().numpy(), " ",
                      index3.data.cpu().numpy())

            test_loss = mim.item()

            if max_loss > test_loss:
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


def show_mnist(first_image):
    pixels = first_image.reshape((28, 28))
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