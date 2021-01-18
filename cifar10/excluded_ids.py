from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.utils.prune as prune

import sys

import argparse
import os

import cifar10_utils
#from Independent_Net import IndependentNet
from Bin_Cifar10_Net import BinCifar10Net
#from AlexNet import AlexNet

from image_utils import *
import random

from torchvision.utils import make_grid
import matplotlib
import pickle

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(sci_mode=False)

EPS = sys.float_info.epsilon

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None

class_numbers = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 0: 0}

transformations_dict = {0: "original",
                       1: "scale",
                       2: "rotate",
                       3: "reverse pixel value",
                       4: "sobel total",
                       5: "sobel x",
                       6: "sobel y",
                       7: "gaussian blur",
                       8: "randcom_crop_upscale_gauss_blur",
                       9: "randcom_crop_upscale_sobel_total",
                       10: "randcom_crop_upscale_rev_pixels",
                       11: "no_jitter_rotate(image, -46)",
                       12: "randcom_crop_upscale(image, 20",
                       13: "randcom_crop_upscale(image, 22",
                       14: "randcom_crop_upscale(image, 26",
                       15: "no_jitter_random_corpse(image, 22",
                       16: "randcom_crop_upscale(image, 18",
                       17: "random_erase",
                       18: "noise",
                       19: "image_1",
                       20: "image_2",
                       21: "image_3",
                       22: "image_4"}



def save_images(images, transformation):
    print(transformations_dict[transformation])
    numpy_cluster = images.cpu().detach()
    save_cluster(numpy_cluster, transformations_dict[transformation], 0)











def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def exclude():
    global first
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = cifar10_utils.load_cifar10(cifar10_utils.CIFAR10_FOLDER)
    X_train, y_train, X_test, targets = cifar10_utils.preprocess_cifar10_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw)

    X_train = torch.from_numpy(X_train)
    X_train /= 255

    excluded = {}
    all_excluded = []
    for idx, i in enumerate(y_train):
        if i not in excluded.keys():
            ex = []
            ex.append(idx)
            all_excluded.append(idx)
            excluded[i] = ex
        else:
            if len(excluded[i]) == 1000:
                continue

            excluded[i].append(idx)
            all_excluded.append(idx)

    to_train = [x for x in range(50000) if x not in all_excluded]

    query_images = X_train[all_excluded]
    print("query images: ", query_images.shape)
    query_targets = y_train[all_excluded]
    print("query targets: ", query_targets.shape)
    X_train = X_train[to_train, :]
    y_train = y_train[to_train]
    print("y_train: ", y_train.shape)
    print("X_train: ", X_train.shape)

    return X_train, y_train, query_images, query_targets
    ###############################################


def to_tensor(X):
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


def main():
    """
    Main function
    """
    # Run the training operation
    exclude()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    FLAGS, unparsed = parser.parse_known_args()

    main()