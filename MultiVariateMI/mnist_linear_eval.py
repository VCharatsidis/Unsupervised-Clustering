from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import os
import torch

from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from mnist_linear_net import MnistLinearNet
# Default constants
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 30000
BATCH_SIZE_DEFAULT = 160
EVAL_FREQ_DEFAULT = 20
LINEAR_INPUT = 256
FLAGS = None
EPS=sys.float_info.epsilon

encoder = torch.load("MNIST_solver.model")
encoder.eval()


def flatten(out):
    return out.view(out.shape[0], -1)


def multi_filter_flatten(out):
    return out.view(out.shape[0], out.shape[1], -1)


def forward_block(X, ids, classifier, optimizer, train, targets):
    x_train = X[ids, :]
    x_tensor = to_Tensor(x_train)
    images = x_tensor / 255
    images = images.to('cuda')

    p1 = torch.zeros([BATCH_SIZE_DEFAULT, 10]).to('cuda')
    p2 = torch.zeros([BATCH_SIZE_DEFAULT, 10]).to('cuda')
    p3 = torch.zeros([BATCH_SIZE_DEFAULT, 10]).to('cuda')

    _, encodings = encoder(images, p1, p2, p3)
    preds = classifier(encodings.detach())

    desired_array = [int(numeric_string) for numeric_string in targets[ids]]

    tensor_targets = torch.LongTensor(desired_array).unsqueeze(dim=1).to('cuda')
    y_onehot = torch.FloatTensor(BATCH_SIZE_DEFAULT, 10).to('cuda')
    y_onehot.zero_()
    y_onehot.scatter_(1, tensor_targets, 1)

    cross_entropy_loss = - (y_onehot * torch.log(preds)).sum(dim=1).mean()

    if train:
        for p in encoder.parameters():
            p.requires_grad = False

        optimizer.zero_grad()
        cross_entropy_loss.backward(retain_graph=True)
        optimizer.step()

    return preds, cross_entropy_loss


def print_params(model):
    for param in model.parameters():
        print(param.data)


def train():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    targets = mnist.target[60000:]
    train_targets = mnist.target[:60000]

    X_train = mnist.data[:60000]
    X_test = mnist.data[60000:]

    print(X_test.shape)

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'colons\\colon_' + str(0) + '.model'
    predictor_model = os.path.join(script_directory, filepath)

    linear_classifier = MnistLinearNet(LINEAR_INPUT).to('cuda')
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=LEARNING_RATE_DEFAULT)

    max_loss = 1999
    max_loss_iter = 0
    PATIENCE = 50
    patience = 0
    best_accuracy = 0

    for iteration in range(MAX_STEPS_DEFAULT):
        linear_classifier.train()
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        train = True
        preds, cross_entropy_loss = forward_block(X_train, ids, linear_classifier, optimizer, train, train_targets)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            linear_classifier.eval()
            print()
            print("==============================================================")
            print("ITERATION: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best acc: ", max_loss_iter,
                  ": ", best_accuracy)
            print("patience: ", patience)

            print()
            loss, acc = measure_acc_augments(X_test, linear_classifier, targets)

            if best_accuracy < acc:
                best_accuracy = acc
                max_loss_iter = iteration
                patience = 0
                print("models saved iter: " + str(iteration))
            else:
                patience += 1

            if patience > PATIENCE:
                print("For ", patience, " iterations we do not have a better accuracy")
                print("best accuracy: ", best_accuracy, " at iter: ", max_loss_iter)
                print("accuracy at stop: ", acc, "loss at stop: ", loss)

                file = open("experiments.txt", "a")

                accuracy_info = ",  BEST ACCURACY: " + str(best_accuracy) + " at iter: " + str(max_loss_iter) + "\n"
                file.write(accuracy_info)
                file.close()
                break


def measure_acc_augments(X_test, classifier, targets):
    runs = len(X_test) // BATCH_SIZE_DEFAULT
    avg_loss = 0
    sum_correct = 0

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        test_ids = np.array(test_ids)
        optimizer = []
        preds, mim = forward_block(X_test, test_ids, classifier, optimizer, False, targets)

        desired_array = [int(numeric_string) for numeric_string in targets[test_ids]]
        sum_correct += accuracy(preds, desired_array)
        avg_loss += mim.item()

    average_test_loss = avg_loss / runs
    accuracy_test_set = sum_correct / (runs * BATCH_SIZE_DEFAULT)
    print("Test set avg loss: ", average_test_loss, " avg accuracy: ", accuracy_test_set)

    return average_test_loss, accuracy_test_set


def accuracy(predictions, targets):
   predictions = predictions.cpu().detach().numpy()
   preds = np.argmax(predictions, 1)
   result = preds == targets
   sum = np.sum(result)
   #accur = sum / float(targets.shape[0])

   return sum


def miss_classifications(cluster):
    mfe = most_frequent(cluster)
    missclassifications = 0
    for j in cluster:
        if j != mfe:
            missclassifications += 1

    return missclassifications


def most_frequent(List):
    return max(set(List), key=List.count)


def to_Tensor(X):
    X = np.reshape(X, (BATCH_SIZE_DEFAULT, 1, 28, 28))
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