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

import random
from encoderSpecialisation import Specialist
from stl_utils import rotate, scale, to_gray, random_erease, vertical_flip


# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 40
EVAL_FREQ_DEFAULT = 100
NUMBER_CLASSES = 1
FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def encode_4_patches(image, colons,
                     p1=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p2=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p3=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p4=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p5=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p6=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p7=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p8=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p9=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     p0=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                     ):

    #split_at_pixel = 19

    # image = np.reshape(image, (BATCH_SIZE_DEFAULT, 1, 28, 28))
    # image = torch.FloatTensor(image)

    # print(image.shape)
    # print(image.shape[2])
    # print(image.shape[3])
    # input()
    width = image.shape[2]
    height = image.shape[3]

    split_at_pixel = 50

    image = image[:, :, 20:70, 20:70]
    # image_1 = image[:, :, 0: split_at_pixel, 0: split_at_pixel]
    # image_2 = image[:, :, 85 - split_at_pixel:, 0: split_at_pixel]
    # image_3 = image[:, :, 0: split_at_pixel, 0: split_at_pixel]
    # image_4 = image[:, :, 85 - split_at_pixel:, 85 - split_at_pixel:]

    # patches = {0: image_1,
    #            1: image_2,
    #            2: image_3,
    #            3: image_4}
    #
    # patch_ids = np.random.choice(len(patches), size=4, replace=False)
    #
    # augments = {0: to_gray(patches[patch_ids[0]], BATCH_SIZE_DEFAULT),
    #             1: rotate(patches[patch_ids[1]], 20, BATCH_SIZE_DEFAULT),
    #             2: rotate(patches[patch_ids[2]], -20, BATCH_SIZE_DEFAULT),
    #             3: scale(patches[patch_ids[3]], 40, 5, BATCH_SIZE_DEFAULT),
    #             4: vertical_flip(patches[patch_ids[0]], BATCH_SIZE_DEFAULT),
    #             5: scale(patches[patch_ids[1]], 30, 10, BATCH_SIZE_DEFAULT),
    #             6: random_erease(patches[patch_ids[2]], BATCH_SIZE_DEFAULT),
    #             7: patches[patch_ids[3]]}

    # augments = {0: to_gray(image, 3, BATCH_SIZE_DEFAULT),
    #             1: rotate(image, 20, BATCH_SIZE_DEFAULT),
    #             2: rotate(image, -20, BATCH_SIZE_DEFAULT),
    #             3: scale(image, 40, 5, BATCH_SIZE_DEFAULT),
    #             4: vertical_flip(image, BATCH_SIZE_DEFAULT),
    #             5: scale(image, 30, 10, BATCH_SIZE_DEFAULT),
    #             6: random_erease(image, BATCH_SIZE_DEFAULT),
    #             7: image}
    #
    # ids = np.random.choice(len(augments), size=1, replace=False)
    #
    # image_2 = augments[ids[0]]

    image = image.to('cuda')
    #image_2 = image_2.to('cuda')

    # image = torch.transpose(image, 1, 3)
    # show_mnist(image[0], image[0].shape[1], image[0].shape[2])
    # image = torch.transpose(image, 1, 3)
    #
    # image_2 = torch.transpose(image_2, 1, 3)
    # show_mnist(image_2[0], image_2[0].shape[1], image_2[0].shape[2])
    # image_2 = torch.transpose(image_2, 1, 3)
    #
    # image_3 = torch.transpose(image_3, 1, 3)
    # show_mnist(image_3[0], image_3[0].shape[1], image_3[0].shape[2])
    # image_3 = torch.transpose(image_3, 1, 3)
    #
    # image_1 = torch.transpose(image_1, 1, 3)
    # show_mnist(image_1[0], image_1[0].shape[1], image_1[0].shape[2])
    # image_1 = torch.transpose(image_1, 1, 3)

    p1 = p1.cuda()
    p2 = p2.cuda()
    p3 = p3.cuda()
    p4 = p4.cuda()
    p5 = p5.cuda()
    p6 = p6.cuda()
    p7 = p7.cuda()
    p8 = p8.cuda()
    p9 = p9.cuda()
    p0 = p0.cuda()

    new_preds = []
    for idx, colon in enumerate(colons):

        # colons[idx] = colons[idx].to('cuda')

        pred_1 = colons[idx](image, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
        #pred_2 = colons[idx](image_2)
        pred = pred_1 #* pred_2

        new_preds.append(pred.to('cpu'))

    products = []
    for prod in range(10):
        product = torch.ones([BATCH_SIZE_DEFAULT, 1])

        for idx, prediction in enumerate(new_preds):
            if idx != prod:
                product *= torch.ones(prediction.shape) - prediction
            else:
                product *= prediction

        products.append(product)


    # print(len(products))
    # print(products[0])
    # print(products[0].shape)
    #
    # input()
    # image_1 = image
    # image_2 = random_erease(image, BATCH_SIZE_DEFAULT)
    # image_2 = torch.transpose(image_2, 1, 3)
    # print(image_2.shape)
    # show_mnist(image_2[0], image_2[0].shape[1], image_2[0].shape[2])
    #
    # image_3 = scale(image, BATCH_SIZE_DEFAULT)
    # show_mnist(image_3[0], image_3[0].shape[1], image_3[0].shape[2])
    #
    # image_4 = random_erease(image, BATCH_SIZE_DEFAULT)
    # show_mnist(image_4[0], image_4.shape[1], image_4.shape[2])

    return products, new_preds


def forward_block(X, ids, colons, optimizers, train, to_tensor_size,
                  p1=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p2=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p3=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p4=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p5=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p6=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p7=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p8=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p9=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  p0=torch.zeros([BATCH_SIZE_DEFAULT, NUMBER_CLASSES]),
                  ):

    x_train = X[ids, :]

    x_tensor = to_tensor(x_train, to_tensor_size)

    images = x_tensor/255.0

    products, new_preds = encode_4_patches(images, colons, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)

    losses = []
    total_loss = torch.zeros([1])

    for p in products:
        mean = p.mean(dim=0)
        log_p = -torch.log(mean)
        losses.append(log_p)
        total_loss += log_p

    if train:
        torch.autograd.set_detect_anomaly(True)

        for idx, i in enumerate(optimizers):
            i.zero_grad()

        total_loss.backward(retain_graph=True)

        for idx, i in enumerate(optimizers):
            i.step()

    return products, total_loss, new_preds


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

    input = 2058
    #input = 1152


    c = Specialist(3, input)
    c = c.cuda()
    colons.append(c)

    c2 = Specialist(3, input)
    c2 = c2.cuda()
    colons.append(c2)

    c3 = Specialist(3, input)
    c3.cuda()
    colons.append(c3)

    c4 = Specialist(3, input)
    c4.cuda()
    colons.append(c4)

    c5 = Specialist(3, input)
    c5.cuda()
    colons.append(c5)

    c6 = Specialist(3, input)
    c6.cuda()
    colons.append(c6)

    c7 = Specialist(3, input)
    c7.cuda()
    colons.append(c7)

    c8 = Specialist(3, input)
    c8.cuda()
    colons.append(c8)

    c9 = Specialist(3, input)
    c9.cuda()
    colons.append(c9)

    c0 = Specialist(3, input)
    c0.cuda()
    colons.append(c0)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)

    optimizer2 = torch.optim.Adam(c2.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer2)

    optimizer3 = torch.optim.Adam(c3.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer3)

    optimizer4 = torch.optim.Adam(c4.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer4)

    optimizer5 = torch.optim.Adam(c5.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer5)

    optimizer6 = torch.optim.Adam(c6.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer6)

    optimizer7 = torch.optim.Adam(c7.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer7)

    optimizer8 = torch.optim.Adam(c8.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer8)

    optimizer9 = torch.optim.Adam(c9.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer9)

    optimizer0 = torch.optim.Adam(c0.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer0)

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        products, mim, new_preds = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p0 = new_preds
        products, mim, new_preds = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p0 = new_preds
        products, mim, new_preds = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)

        if iteration % EVAL_FREQ_DEFAULT == 0:

            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)

            products, mim, new_preds = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)
            print("loss 1: ", mim.item())
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p0 = new_preds
            products, mim, new_preds = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
            print("loss 2: ", mim.item())
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p0 = new_preds
            products, mim, new_preds = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
            print("loss 3: ", mim.item())

            print()
            print("iteration: ", iteration)

            print_info(products, targets, test_ids)

            test_loss = mim.item()

            if max_loss > test_loss:
                max_loss = test_loss
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


def print_info(products, targets, test_ids):
    #print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 10: ""}
    # print(products[0].shape)
    # input()
    for i in range(products[0].shape[0]):
        res = ""

        for specialist in products:

            index = torch.round(specialist[i])

            res += str(int(index.data.cpu().numpy()[0])) + " "

        res += ", "

        label = targets[test_ids[i]]
        print_dict[label] += res

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