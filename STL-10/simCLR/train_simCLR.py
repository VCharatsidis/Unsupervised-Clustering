from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import argparse
import os
from stl10_input import read_all_images, read_labels
from simCLR_NET import SimCLRNet
from stl_utils import *
import random
import sys
from torchvision.utils import make_grid
import matplotlib
from torchvision import models

from pytorch_metric_learning.losses import NTXentLoss

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()


#EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000

BATCH_SIZE_DEFAULT = 140

INPUT_NET = 4608
SIZE = 32
NETS = 1

DROPOUT = [0.0, 0.0, 0.0, 0.0, 0.0]
DESCRIPTION = " Image size: "+str(SIZE) + " , Dropout2d: "+str(DROPOUT)

EVAL_FREQ_DEFAULT = 1000
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None

loss = NTXentLoss(temperature=0.1)


labels_to_imags = {1: "airplane",
                   2: "bird    ",
                   3: "car     ",
                   4: "cat     ",
                   5: "deer    ",
                   6: "dog     ",
                   7: "horse   ",
                   8: "monkey  ",
                   9: "ship    ",
                   0: "truck   "}

images_to_labels ={"airplane": 1,
                   "bird    ": 2,
                   "car     ": 3,
                   "cat     ": 4,
                   "deer    ": 5,
                   "dog     ": 6,
                   "horse   ": 7,
                   "monkey  ": 8,
                   "ship    ": 9,
                   "truck   ": 0}


def get_targets(number_classes):
    return (torch.ones([number_classes]).to('cuda')) / number_classes


def encode_4_patches(image, encoder):
    pad = (96 - SIZE) // 2
    image /= 255

    crop_size = 56
    crop_pad = (96 - crop_size) // 2
    crop_preparation = scale(image, crop_size, crop_pad, BATCH_SIZE_DEFAULT)
    crop_preparation = crop_preparation[:, :, crop_pad:96 - crop_pad, crop_pad:96 - crop_pad]
    # crop_prep_horizontal = horizontal_flip(crop_preparation)

    crop_size2 = 66
    crop_pad2 = (96 - crop_size2) // 2
    crop_preparation2 = scale(image, crop_size2, crop_pad2, BATCH_SIZE_DEFAULT)
    crop_preparation2 = crop_preparation2[:, :, crop_pad2:96 - crop_pad2, crop_pad2:96 - crop_pad2]
    crop_prep_horizontal2 = horizontal_flip(crop_preparation2)

    horiz_f = horizontal_flip(image, BATCH_SIZE_DEFAULT)

    soft_bin_hf = binary(horiz_f)
    soft_bin_hf = scale(soft_bin_hf, SIZE, pad, BATCH_SIZE_DEFAULT)
    soft_bin_hf = soft_bin_hf[:, :, pad:96 - pad, pad:96 - pad]
    #rev_soft_bin_hf = torch.abs(1 - soft_bin_hf)

    original_image = scale(image, SIZE, pad, BATCH_SIZE_DEFAULT)
    original_image = original_image[:, :, pad:96 - pad, pad:96 - pad]

    original_hfliped = horizontal_flip(original_image, BATCH_SIZE_DEFAULT)

    augments = {
                0: sobel_filter_x(original_hfliped, BATCH_SIZE_DEFAULT),
                1: sobel_filter_y(original_hfliped, BATCH_SIZE_DEFAULT),
                2: sobel_total(original_image, BATCH_SIZE_DEFAULT),
                3: soft_bin_hf,
                4: torch.abs(1 - original_image),
                5: rotate(original_image, 40),
                6: scale(original_hfliped, SIZE - 12, 6, BATCH_SIZE_DEFAULT),
                7: color_jitter(random_crop(crop_preparation, SIZE, BATCH_SIZE_DEFAULT)),
                8: color_jitter(random_crop(crop_prep_horizontal2, SIZE, BATCH_SIZE_DEFAULT)),
                9: original_image
                }

    ids = np.random.choice(len(augments), 2, replace=False)

    image_1 = augments[ids[0]]
    image_2 = augments[ids[1]]

    # show_gray(image_1)
    # show_gray(image_2)

    embeddings1 = encoder(image_1.to('cuda'))
    embeddings2 = encoder(image_2.to('cuda'))

    return embeddings1, embeddings2, original_image, ids


def forward_block(X, ids, encoder, optimizer, train):
    x_train = X[ids, :]
    x_train = rgb2gray(x_train)

    x_tensor = to_tensor(x_train)
    x_tensor = x_tensor.unsqueeze(0)
    images = x_tensor.transpose(0, 1)
    images = images.transpose(2, 3)

    embeddings1, embeddings2, orig_image, aug_ids = encode_4_patches(images, encoder)

    embeddings = torch.cat((embeddings1, embeddings2))
    indices = torch.arange(0, embeddings1.size(0), device=embeddings1.device)
    labels = torch.cat((indices, indices))
    loss_a = loss(embeddings, labels)

    total_loss = loss_a

    if train:
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

    return embeddings1, embeddings2, total_loss,  orig_image, aug_ids


def batch_entropy(pred, targets):
    batch_mean_preds = pred.mean(dim=0)
    H_batch = torch.log(batch_mean_preds).mean()

    return H_batch


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images/iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def measure_acc_augments(X_test, colons):
    runs = len(X_test)//BATCH_SIZE_DEFAULT
    sum_loss = 0

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        optimizers = []
        p1, p2, mim, orig_image, aug_ids = forward_block(X_test, test_ids, colons, optimizers, False)

        sum_loss += mim.item()

    print()
    print("AUGMENTS avg loss: ", sum_loss / runs)
    print()

    return sum_loss/runs


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

    train_path = "..\\data_2\\stl10_binary\\unlabeled_X.bin"

    print(train_path)
    X_train = read_all_images(train_path)

    # train_y_File = "..\\data\\stl10_binary\\train_y.bin"
    # y_train = read_labels(train_y_File)

    ########### test ##############################
    testFile = "..\\data\\stl10_binary\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "..\\data\\stl10_binary\\test_y.bin"
    targets = read_labels(test_y_File)

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = '..\\Linear_classifier\\models\\simCLR' + '.model'
    loss_net_path = os.path.join(script_directory, filepath)

    encoder = SimCLRNet(1, INPUT_NET, DROPOUT).to('cuda')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    max_loss = 1999
    max_loss_iter = 0

    print(encoder)
    print("X_train: ", X_train.shape, " X_test: ", X_test.shape, " targets: ", targets.shape)

    # labels_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    # for idx, i in enumerate(y_train):
    #     labels_dict[i].append(idx)

    test_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    for idx, i in enumerate(targets):
        test_dict[i].append(idx)

    for iteration in range(MAX_STEPS_DEFAULT):
        encoder.train()

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        embeddings1, embeddings2, total_loss, orig_image, aug_ids = forward_block(X_train, ids, encoder, optimizer, train)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            encoder.eval()
            print("==================================================================================")
            print("ITERATION: ", iteration,
                  ",  batch size: ", BATCH_SIZE_DEFAULT,
                  ",  lr: ", LEARNING_RATE_DEFAULT,
                  ",  best loss iter: ", max_loss_iter,
                  "-", max_loss,
                  ",", DESCRIPTION)

            loss = measure_acc_augments(X_test, encoder)

            if max_loss > loss:
                max_loss = loss
                max_loss_iter = iteration

                print("models saved iter: " + str(iteration))
                torch.save(encoder, loss_net_path)


def to_tensor(X):
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


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