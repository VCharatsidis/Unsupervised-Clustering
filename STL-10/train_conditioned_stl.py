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
from socialSTL_encoder import SocialEncoderSTL
import copy

from torchvision import transforms
import torchvision.transforms.functional as F
import copy
import numpy as np
from torch.autograd import Variable
import torch

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 130
EVAL_FREQ_DEFAULT = 200

FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def encode_4_patches(image, colons,
                     p1=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p2=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p3=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p4=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p5=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p6=torch.zeros([BATCH_SIZE_DEFAULT, 10])
                     ):

    split_at_pixel = 30
    #split_at_pixel = 19

    # image = np.reshape(image, (BATCH_SIZE_DEFAULT, 1, 28, 28))
    # image = torch.FloatTensor(image)

    # print(image.shape)
    # print(image.shape[2])
    # print(image.shape[3])
    # input()
    width = image.shape[2]
    height = image.shape[3]

    # image_1 = image[:, :, 0: split_at_pixel, :]
    # image_2 = image[:, :, width - split_at_pixel:, :]
    # image_3 = image[:, :, :, 0: split_at_pixel]
    # image_4 = image[:, :, :, height - split_at_pixel:]

    # image_1 = image[:, :, 10: 30, 5:35]
    # image_2 = image[:, :, 45: 75, 5:35]
    #
    # image_3 = image[:, :, 15: 45, 35:65]
    # image_4 = image[:, :, 45: 75, 35:65]
    #
    # image_5 = image[:, :, 25: 55, 55:85]
    # image_6 = image[:, :, 55: 85, 55:85]

    # image_1 = image[:, :, 0: 50, 0:50]
    # image_2 = image[:, :, 40:90, 40:90]
    # image_3 = image[:, :, 40:90, 0: 50]
    # image_4 = image[:, :, 20:70, 20:70]
    # image_5 = image[:, :, 10:60, 10:60]
    # image_6 = image[:, :, 30:80, 30:80]

    image_1 = image[:, :, 0: 45, 0:30]
    image_2 = image[:, :, 45:90, 0:30]
    image_2 = to_gray(image_2)

    image_3 = image[:, :, 0: 45, 30:60]
    image_3 = scale(image_3)

    image_4 = image[:, :, 45:90, 30:60]
    image_5 = image[:, :, 0: 45, 60:90]
    image_6 = image[:, :, 45:90, 60:90]
    image_6 = to_gray(image_6)

    image_1 = image_1.to('cuda')
    image_2 = image_2.to('cuda')
    image_3 = image_3.to('cuda')
    image_4 = image_4.to('cuda')
    image_5 = image_5.to('cuda')
    image_6 = image_6.to('cuda')

    p1 = p1.to('cuda')
    p2 = p2.to('cuda')
    p3 = p3.to('cuda')
    p4 = p4.to('cuda')
    p5 = p5.to('cuda')
    p6 = p6.to('cuda')

    pred_1 = colons[0](image_1, p2, p3, p4, p5, p6)
    pred_2 = colons[0](image_2, p1, p3, p4, p5, p6)
    pred_3 = colons[0](image_3, p1, p2, p4, p5, p6)
    pred_4 = colons[0](image_4, p1, p2, p3, p5, p6)
    pred_5 = colons[0](image_5, p1, p2, p3, p4, p6)
    pred_6 = colons[0](image_6, p1, p2, p3, p4, p5)

    return pred_1, pred_2, pred_3, pred_4, pred_5, pred_6


def scale(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))
    size = 20
    pad = 5

    for i in range(X_copy.shape[0]):
        transformation = transforms.Resize((45, 30), interpolation=2)
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def to_gray(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        transformation = transforms.Grayscale(3)
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def forward_block(X, ids, colons, optimizers, train, to_tensor_size,
                     p1=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p2=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p3=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p4=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p5=torch.zeros([BATCH_SIZE_DEFAULT, 10]),
                     p6=torch.zeros([BATCH_SIZE_DEFAULT, 10])):

    x_train = X[ids, :]

    x_tensor = to_tensor(x_train, to_tensor_size)

    images = x_tensor/255.0

    pred_1, pred_2, pred_3, pred_4, pred_5, pred_6 = encode_4_patches(images, colons, p1, p2, p3, p4, p5, p6)

    product = pred_1 * pred_2 * pred_3 * pred_4 * pred_5 * pred_6
    product = product.mean(dim=0)
    log_product = torch.log(product)

    loss = - log_product.mean(dim=0)

    if train:
        torch.autograd.set_detect_anomaly(True)

        for i in optimizers:
            i.zero_grad()

        loss.backward(retain_graph=True)

        for i in optimizers:
            i.step()

    return pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, loss


def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():
    #
    fileName = "data\\stl10_binary\\train_X.bin"
    X_train = read_all_images(fileName)

    testFile = "data\\stl10_binary\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "data\\stl10_binary\\test_y.bin"
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

    preds = 50
    input = 1170
    #input = 1152

    # c = Ensemble()
    # c.cuda()

    c = SocialEncoderSTL(3, input)
    c.cuda()
    colons.append(c)

    # c2 = SocialEncoderSTL(3, input)
    # c2.cuda()
    # colons.append(c2)
    #
    # c3 = SocialEncoderSTL(3, input)
    # c3.cuda()
    # colons.append(c3)
    #
    # c4 = SocialEncoderSTL(3, input)
    # c4.cuda()
    # colons.append(c4)

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

    max_loss = 1999

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, p3, p4, p5, p6, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)
        p1, p2, p3, p4, p5, p6, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3, p4, p5, p6)
        p1, p2, p3, p4, p5, p6, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3, p4, p5, p6)


        if iteration % EVAL_FREQ_DEFAULT == 0:

            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, p3, p4, p5, p6, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)
            p1, p2, p3, p4, p5, p6, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3,
                                               p4, p5, p6)
            p1, p2, p3, p4, p5, p6, mim = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, p1, p2, p3,
                                               p4, p5, p6)

            print()
            print("iteration: ", iteration)

            print(p1[0])
            print(p2[0])
            print(p3[0])
            print(p4[0])
            print(p5[0])
            print(p6[0])

            print_info(p1, p2, p3, p4, p5, p6, targets, test_ids)

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
    pixels = first_image.reshape((w, h))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def print_info(p1, p2, p3, p4, p5, p6, targets, test_ids):
    #print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 10: ""}
    for i in range(p1.shape[0]):
        if i == 10:
            print("")

        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)
        val, index3 = torch.max(p3[i], 0)
        val, index4 = torch.max(p4[i], 0)
        val, index5 = torch.max(p5[i], 0)
        val, index6 = torch.max(p6[i], 0)

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + " " + \
                 str(index3.data.cpu().numpy()) + " " + str(index4.data.cpu().numpy()) + " " + \
                 str(index5.data.cpu().numpy()) + " " + str(index6.data.cpu().numpy()) + " , "

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