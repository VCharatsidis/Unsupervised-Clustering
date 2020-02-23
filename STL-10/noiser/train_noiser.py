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

from encoder_model import EncoderNet
from noiser_model import NoiserNet
from metaModel import MetaModel
from loss import entropy_balance_loss
from stl_utils import rotate, scale, to_grayscale, random_erease, vertical_flip, horizontal_flip, sobel_filter_y, sobel_filter_x, sobel_total
import random
import matplotlib
from torchvision.utils import make_grid
import sys



# Default constants
EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000

BATCH_SIZE_DEFAULT = 128
INPUT_NET = 8192

EVAL_FREQ_DEFAULT = 100
NUMBER_CLASSES = 1

FLAGS = None


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def kl_divergence(p, q):
    return torch.nn.functional.kl_div(p, q)


def rgb2gray(rgb):

    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def show_gray_numpy(image_1):
    z = image_1.squeeze(1)
    pixels = z[0]
    plt.imshow(pixels, cmap='gray')
    plt.show()

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


def forward_block(X, ids, colons, optimizers, train, to_tensor_size, measure):

    x_train = X[ids, :]
    x_train = rgb2gray(x_train)

    x_tensor = to_tensor(x_train, to_tensor_size)
    x_tensor = x_tensor.unsqueeze(0)
    images = x_tensor.transpose(0, 1)
    images = images.transpose(2, 3)

    #show_gray(images[0])

    images /= 255

    size = 40
    pad = (96 - size) // 2
    original_image = scale(images, size, pad, BATCH_SIZE_DEFAULT)
    # show_gray(original_image)

    original_image = original_image[:, :, pad:96 - pad, pad:96 - pad]
    # show_gray(original_image)

    augments = {0: horizontal_flip(original_image, BATCH_SIZE_DEFAULT),
                1: original_image,
                2: vertical_flip(original_image, BATCH_SIZE_DEFAULT),
                3: scale(original_image, size - 8, 4, BATCH_SIZE_DEFAULT)}

    ids = np.random.choice(len(augments), size=4, replace=False)

    image_1 = colons["noiser"](augments[ids[0]]).view(BATCH_SIZE_DEFAULT, 1, size, size) * augments[ids[0]]
    # print(original_image[0].sum())
    # print(original_image[0][0])
    image_2 = colons["noiser"](augments[ids[1]]).view(BATCH_SIZE_DEFAULT, 1, size, size) * augments[ids[1]]
    # print(image_2[0].sum())
    # print(image_2[0][0])
    #input()

    image_1 = image_1.to('cuda')
    image_2 = image_2.to('cuda')

    preds_1 = colons["encoder 1"](image_1)
    preds_2 = colons["encoder 2"](image_2)

    # entropy_loss_1 = entropy_balance_loss(preds_1)
    # entropy_loss_2 = entropy_balance_loss(preds_2)

    mean_prediction = (preds_1 + preds_2)/2
    H = - (mean_prediction * torch.log(mean_prediction)).sum(dim=1).mean(dim=0)

    batch_mean_preds = mean_prediction.mean(dim=0)
    H_batch = (torch.log(batch_mean_preds)).sum()

    mean_entropy_loss = H - H_batch

    #mean_entropy_loss = entropy_balance_loss(mean_prediction)

    # product = preds_1 * preds_2
    # product = product.mean(dim=0)
    # log_product = torch.log(product)
    # diversion_loss = - log_product.sum(dim=0)

    noise_loss = image_1.sum(dim=3).sum(dim=2).sum(dim=1).mean() + image_2.sum(dim=3).sum(dim=2).sum(dim=1).mean()
    noiser_loss = noise_loss + mean_entropy_loss

    # predictions = torch.cat([preds_1, preds_2], 1)
    # meta_prediction = colons["meta"](predictions.detach())
    # squared = meta_prediction * meta_prediction
    # product = squared.mean(dim=0)
    # log_product = torch.log(product)
    # meta_loss = - log_product.mean(dim=0)

    #meta_loss = entropy_balance_loss(meta_prediction)

    if train:
        torch.autograd.set_detect_anomaly(True)

        # optimizers["encoder 1"].zero_grad()
        # entropy_loss_1.backward(retain_graph=True)
        # optimizers["encoder 1"].step()
        #
        # optimizers["encoder 2"].zero_grad()
        # entropy_loss_2.backward(retain_graph=True)
        # optimizers["encoder 2"].step()
        #
        # optimizers["meta"].zero_grad()
        # meta_loss.backward()
        # optimizers["meta"].step()

        optimizers["encoder 1"].zero_grad()
        optimizers["encoder 2"].zero_grad()
        mean_entropy_loss.backward(retain_graph=True)
        optimizers["encoder 1"].step()
        optimizers["encoder 2"].step()

        optimizers["noiser"].zero_grad()
        noiser_loss.backward()
        optimizers["noiser"].step()

    return preds_1, preds_2, noiser_loss, original_image, image_1, image_2, mean_prediction


def entropies_loss(pred, coeff):
    return pred_entropy(pred) - coeff * batch_entropy(pred)


def pred_entropy(prediction):
    prediction[(prediction < EPS).data] = EPS
    pred = prediction
    H = - (pred * torch.log(pred)).sum(dim=1).mean(dim=0)
    return H


def batch_entropy(pred):
    batch_mean_preds = pred.mean(dim=0)
    H_batch = - (batch_mean_preds * torch.log(batch_mean_preds)).sum()
    # print((batch_mean_preds * torch.log(batch_mean_preds)).sum())
    # print((0.1 * torch.log(batch_mean_preds)).sum())
    # print()
    # input()
    # H_batch = (torch.log(batch_mean_preds)).mean()

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


def measure_acc_augments(X_test, colons, targets, noiser):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    runs = len(X_test)//BATCH_SIZE_DEFAULT
    avg_loss = 0

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        optimizers = []
        p1, p2, mim, original_image, image_1, image_2, meta_prediction = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, True)

        avg_loss += mim.item()
        for i in range(p1.shape[0]):
            val, index = torch.max(p1[i], 0)
            val, index2 = torch.max(p2[i], 0)

            preds = [index.data.cpu().numpy(), index2.data.cpu().numpy()]
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

    colons = {}

    optimizers = {}
    colons_paths = []

    filepath = 'encoders\\encoder_' + str(0) + '.model'
    predictor_model = os.path.join(script_directory, filepath)
    colons_paths.append(predictor_model)

    c1 = EncoderNet(1, INPUT_NET)
    c1 = c1.cuda()
    colons["encoder 1"] = c1
    optimizers["encoder 1"] = torch.optim.Adam(c1.parameters(), lr=LEARNING_RATE_DEFAULT)

    c2 = EncoderNet(1, INPUT_NET)
    c2 = c2.cuda()
    colons["encoder 2"] = c2
    optimizers["encoder 2"] = torch.optim.Adam(c2.parameters(), lr=LEARNING_RATE_DEFAULT)

    # c3 = EncoderNet(1, INPUT_NET)
    # c3 = c3.cuda()
    # colons.append(c3)
    # optimizers["encoder 3"] = torch.optim.Adam(c3.parameters(), lr=LEARNING_RATE_DEFAULT)
    #
    # c4 = EncoderNet(1, INPUT_NET)
    # c4 = c4.cuda()
    # colons.append(c4)
    # optimizers["encoder 4"] = torch.optim.Adam(c4.parameters(), lr=LEARNING_RATE_DEFAULT)

    meta_model = MetaModel(20)
    meta_model = meta_model.cuda()
    colons["meta"] = meta_model
    optimizers["meta"] = torch.optim.Adam(meta_model.parameters(), lr=LEARNING_RATE_DEFAULT)

    noiser = NoiserNet(1, 9792, 40, 40)
    colons["noiser"] = noiser
    optimizers["noiser"] = torch.optim.Adam(noiser.parameters(), lr=LEARNING_RATE_DEFAULT)

    max_loss = 1000000000
    max_loss_iter = 0
    description = "4 augments"

    for iteration in range(MAX_STEPS_DEFAULT):

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        p1, p2, mim, original_image, image_1, image_2, meta_prediction = forward_block(X_train, ids, colons, optimizers, train, BATCH_SIZE_DEFAULT, False)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            print()
            p1, p2, mim, original_image, image_1, image_2, meta_prediction = forward_block(X_test, test_ids, colons, optimizers, False, BATCH_SIZE_DEFAULT, False)
            print("loss 1: ", mim.item())

            print("iteration: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best loss: ", max_loss_iter,
                  ": ", max_loss)
            print("description: ", description)


            print_info(meta_prediction, targets, test_ids)

            test_loss = mim.item()

            if max_loss > test_loss:

                if iteration > -1:
                    save_image(original_image[0], iteration, "original")
                    save_image(image_1.cpu().detach()[0], iteration, "image_1")
                    save_image(image_2.cpu().detach()[0], iteration, "image_2")

                max_loss = test_loss
                max_loss_iter = iteration
                #measure_accuracy(X_test, colons, targets)

                measure_acc_augments(X_test, colons, targets, noiser)


                print("models saved iter: " + str(iteration))
                # for i in range(number_colons):
                #     torch.save(colons[i], colons_paths[i])

            print("test loss " + str(test_loss))
            print("")


def save_image(original_image, iteration, name):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"noised_images/{name}_iter_{iteration}.png", sample)


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


def print_info(p1, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}
    for i in range(p1.shape[0]):
        val, index = torch.max(p1[i], 0)
        #val, index2 = torch.max(p2[i], 0)

        string = str(index.data.cpu().numpy()) + ", " #+ str(index2.data.cpu().numpy()) + ", "

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