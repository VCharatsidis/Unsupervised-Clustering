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
from brain import Brain
import copy

from Mutual_Information.RandomErase import RandomErasing
from torchvision import transforms
import torchvision.transforms.functional as F
import copy
import numpy as np
from torch.autograd import Variable
import torch
from entropy_balance_loss import entropy_balance_loss
from stl_utils import vertical_flip

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000
BATCH_SIZE_DEFAULT = 128
EVAL_FREQ_DEFAULT = 200
NUMBER_CLASSES = 10
FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def rotate(X, degrees, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    #X_copy = to_Tensor(X_copy, batch_size)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        transformation = transforms.RandomRotation(degrees=[degrees, degrees])
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def scale(X, size, pad, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    # X_copy = to_Tensor(X_copy, batch_size)
    X_copy = Variable(torch.FloatTensor(X_copy))

    # if random.uniform(0, 1) > 0.5:
    #     size = 20
    #     pad = 4

    for i in range(X_copy.shape[0]):
        transformation = transforms.Resize(size, interpolation=2)
        trans = transforms.Compose([transformation, transforms.Pad(pad), transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy

def random_erease(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        transformation = RandomErasing()
        trans = transforms.Compose([transforms.ToTensor(), transformation])
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


def forward_block(X, ids, colons, optimizers, train, to_tensor_size):

    x_train = X[ids, :]

    x_tensor = to_tensor(x_train, to_tensor_size)

    images = x_tensor/255.0

    images = scale(images, 40, 28, BATCH_SIZE_DEFAULT)

    # orig_image = torch.transpose(images, 1, 3)
    # show_mnist(orig_image[0], orig_image[0].shape[1], orig_image[0].shape[2])
    # orig_image = torch.transpose(orig_image, 1, 3)

    images = images[:, :, 28:68, 28:68]

    # orig_image = torch.transpose(images, 1, 3)
    # show_mnist(orig_image[0], orig_image[0].shape[1], orig_image[0].shape[2])
    # orig_image = torch.transpose(orig_image, 1, 3)

    images = images.to('cuda')

    balance_coeff = 2
    mean_preds, preds = colons[0](images, train, optimizers, balance_coeff)

    # product = product_predictions.mean(dim=0)
    # log_product = torch.log(product)
    # loss = - log_product.mean(dim=0)

    loss = 0
    for p in preds:
        # print(p.shape)
        # print(p[0])
        loss += entropy_balance_loss(p, balance_coeff)

    loss /= len(preds)

    #loss = entropy_balance_loss(mean_preds, balance_coeff)

    if train:
        torch.autograd.set_detect_anomaly(True)

        for i in optimizers:
            i.zero_grad()

        loss.backward(retain_graph=True)

        for i in optimizers:
            i.step()

    return mean_preds, loss, preds


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

    c = Brain(3)
    c.cuda()
    colons.append(c)

    optimizer = torch.optim.Adam(c.parameters(), lr=LEARNING_RATE_DEFAULT)
    optimizers.append(optimizer)


    max_loss = 1999
    max_loss_iter = 0
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        product_preds, mim, preds = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:

            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            product_preds, mim, preds = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)

            print(mim.item())

            print()
            print("iteration: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best loss: ", max_loss_iter,
                  ": ", max_loss)


            print("preds 00", preds[0][0].cpu().detach().numpy())
            print("preds 01", preds[1][0].cpu().detach().numpy())
            print("preds 02", preds[2][0].cpu().detach().numpy())
            print("preds 03", preds[3][0].cpu().detach().numpy())
            print("preds 04", preds[4][0].cpu().detach().numpy())
            print("preds 05", preds[5][0].cpu().detach().numpy())
            print("preds 06", preds[6][0].cpu().detach().numpy())
            print("preds 07", preds[7][0].cpu().detach().numpy())
            print("preds 08", preds[8][0].cpu().detach().numpy())
            print("preds 09", preds[9][0].cpu().detach().numpy())
            print("preds 10", preds[10][0].cpu().detach().numpy())
            print("preds 11", preds[11][0].cpu().detach().numpy())
            print("preds 12", preds[12][0].cpu().detach().numpy())
            print("preds 13", preds[13][0].cpu().detach().numpy())
            print("preds 14", preds[14][0].cpu().detach().numpy())
            print("preds 15", preds[15][0].cpu().detach().numpy())
            print("preds 16", preds[16][0].cpu().detach().numpy())
            print("preds 17", preds[17][0].cpu().detach().numpy())
            print("preds 18", preds[18][0].cpu().detach().numpy())
            print("preds 19", preds[19][0].cpu().detach().numpy())
            print("preds 20", preds[20][0].cpu().detach().numpy())
            print("preds 21", preds[21][0].cpu().detach().numpy())
            print("preds 22", preds[22][0].cpu().detach().numpy())
            print("preds 23", preds[23][0].cpu().detach().numpy())
            print("preds 24", preds[24][0].cpu().detach().numpy())

            print()
            print("mean pre", product_preds[0].cpu().detach().numpy())
            print()
            print("batch me", product_preds.mean(dim=0).cpu().detach().numpy())

            print_info(product_preds, targets, test_ids)

            test_loss = mim.item()

            if max_loss > test_loss:
                max_loss = test_loss
                max_loss_iter = iteration
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
    # pixels = first_image.reshape((w, h))
    plt.imshow(first_image)
    plt.show()


def print_info(product_preds, targets, test_ids):
    #print_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": ""}
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 10: ""}
    for i in range(product_preds.shape[0]):
        if i == 10:
            print("")

        val, index = torch.max(product_preds[i], 0)

        string = str(index.data.cpu().numpy()) + " , "

        label = targets[test_ids[i]]
        print_dict[label] += string

    for i in print_dict.keys():
        print(i, " : ", print_dict[i])


def miss_classifications(cluster):
    mfe = most_frequent(cluster)
    missclassifications = 0
    for j in cluster:
        if j != mfe:
            missclassifications += 1

    return missclassifications


def most_frequent(List):
    return max(set(List), key=List.count)


def measure_acc_augments(X_test, colons, targets):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    runs = len(X_test)//BATCH_SIZE_DEFAULT
    avg_loss = 0

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        optimizers = []
        product_preds, mim, predictions = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT)
        avg_loss += mim.item()

        for i in range(product_preds.shape[0]):
            val, index = torch.max(product_preds[i], 0)

            preds = [index.data.cpu().numpy()]

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