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
import torchvision.transforms.functional as F
from PIL import Image
from SupervisedNet import SupervisedNet
from stl_utils import rotate, scale, to_grayscale, random_erease, vertical_flip, horizontal_flip, sobel_filter_y, sobel_filter_x, sobel_total,center_crop, color_jitter
import random
import sys

from torchvision.utils import make_grid
import matplotlib
from linear_net import LinearNet

# Default constants
EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000

BATCH_SIZE_DEFAULT = 120
INPUT_NET = 4608
SIZE = 50
NETS = 1
DESCRIPTION = "Augments: 3 augments then 3 sobels x,y,total. Nets: "+str(NETS) +" net. Loss: total_loss = paired_losses - mean_probs_losses. Image size: " + str(SIZE)

EVAL_FREQ_DEFAULT = 200
NUMBER_CLASSES = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
FLAGS = None

encoder = torch.load("")
encoder.eval()


def encode(image):
    pad = (96 - SIZE) // 2
    original_image = scale(image, SIZE, pad, BATCH_SIZE_DEFAULT)
    original_image = original_image[:, :, pad:96 - pad, pad:96 - pad]

    #show_image(original_image)

    original_image = original_image.to('cuda')
    encoding, _ = encoder(original_image)

    return encoding


def forward_block(X, ids, classifier, optimizer, train, targets):
    images = X[ids, :]
    x_tensor = to_tensor(images, BATCH_SIZE_DEFAULT)

    images = x_tensor / 255.0

    encodings = encode(images)
    preds = classifier(encodings)
    print("preds shape", preds.shape)

    y_onehot = torch.FloatTensor(BATCH_SIZE_DEFAULT, NUMBER_CLASSES).to('cuda')
    y_onehot.zero_()
    y_onehot.scatter_(1, targets[ids], 1)

    print(targets[ids])
    print(y_onehot)

    cross_entropy_loss = - (y_onehot * torch.log(preds + EPS)).mean()

    if train:
        optimizer.zero_grad()
        cross_entropy_loss.backward(retain_graph=True)
        optimizer.step()

    return preds, cross_entropy_loss


def save_image(original_image, iteration, name):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/{name}_iter_{iteration}.png", sample)


def accuracy(predictions, targets):
   predictions = predictions.cpu().detach().numpy()
   preds = np.argmax(predictions, 1)
   result = preds == targets
   sum = np.sum(result)
   accur = sum / float(targets.shape[0])

   return accur


def measure_acc_augments(X_test, classifier, targets):
    runs = len(X_test) // BATCH_SIZE_DEFAULT
    avg_loss = 0
    avg_accuracy = 0

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        test_ids = np.array(test_ids)
        optimizer = []
        preds, mim = forward_block(X_test, test_ids, classifier, optimizer, False, targets)

        avg_accuracy += accuracy(preds, targets[test_ids])
        avg_loss += mim.item()

    print()
    print("AUGMENTS avg loss: ", avg_loss / runs, " avg accuracy: ", avg_accuracy / runs)
    print()

    return avg_loss/runs, avg_accuracy / runs


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
    fileName = "..\\data\\stl10_binary\\train_X.bin"
    X_train = read_all_images(fileName)

    train_y_File = "..\\data\\stl10_binary\\train_y.bin"
    y_train = read_labels(train_y_File)

    testFile = "..\\data\\stl10_binary\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "..\\data\\stl10_binary\\test_y.bin"
    targets = read_labels(test_y_File)

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'encoders\\encoder_' + str(0) + '.model'
    path_to_model = os.path.join(script_directory, filepath)

    linearClassifier = SupervisedNet(3, INPUT_NET).to('cuda')
    optimizer = torch.optim.Adam(linearClassifier.parameters(), lr=LEARNING_RATE_DEFAULT)

    max_loss = 1999
    max_loss_iter = 0

    for iteration in range(MAX_STEPS_DEFAULT):
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        preds, mim = forward_block(X_train, ids, linearClassifier, optimizer, train, y_train)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            print()
            print("iteration: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best loss: ", max_loss_iter,
                  ": ", max_loss)
            print("description: ", DESCRIPTION)

            loss, clusters = measure_acc_augments(X_test, linearClassifier, targets)

            if max_loss > loss:
                max_loss = loss
                max_loss_iter = iteration

                print("models saved iter: " + str(iteration))


def to_tensor(X):
    #X = np.reshape(X, (batch_size, 1, 96, 96))
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