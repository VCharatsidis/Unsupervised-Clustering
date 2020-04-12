from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from stl10_input import read_all_images, read_labels

import sys
from stl_utils import *

from torchvision.utils import make_grid
import matplotlib
from linear_net import LinearNet
from RandomNet import RandomNet
from SupervisedClassifier import SupervisedClassifier

# Default constants

LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000

BATCH_SIZE_DEFAULT = 50
INPUT_NET = 8192
SIZE = 36
NETS = 1
EVAL_FREQ_DEFAULT = 200
PATIENCE = 20

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
FLAGS = None


def forward_block(X, ids, classifier, optimizer, train, targets):
    images = X[ids, :]
    x_tensor = to_tensor(images)

    x_train = rgb2gray(x_tensor)
    x_tensor = to_tensor(x_train)
    x_tensor = x_tensor.unsqueeze(0)
    images = x_tensor.transpose(0, 1)
    images = images.transpose(2, 3)

    images = images / 255.0
    pad = (96 - SIZE) // 2
    original_image = scale(images, SIZE, pad, BATCH_SIZE_DEFAULT)
    original_image = original_image[:, :, pad:96 - pad, pad:96 - pad]
    original_image = original_image.to('cuda')

    _, preds = classifier(original_image)

    tensor_targets = torch.LongTensor(targets[ids]).unsqueeze(dim=1).to('cuda')
    y_onehot = torch.FloatTensor(BATCH_SIZE_DEFAULT, 10).to('cuda')
    y_onehot.zero_()
    y_onehot.scatter_(1, tensor_targets, 1)

    cross_entropy_loss = - (y_onehot * torch.log(preds)).sum(dim=1).mean()

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
   #accur = sum / float(targets.shape[0])

   return sum


def measure_acc_augments(X_test, classifier, targets):
    runs = len(X_test) // BATCH_SIZE_DEFAULT
    avg_loss = 0
    sum_correct = 0

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        test_ids = np.array(test_ids)
        optimizer = []
        preds, mim = forward_block(X_test, test_ids, classifier, optimizer, False, targets)

        sum_correct += accuracy(preds, targets[test_ids])
        avg_loss += mim.item()

    average_test_loss = avg_loss / runs
    accuracy_test_set = sum_correct / (runs * BATCH_SIZE_DEFAULT)
    print("Test set avg loss: ", average_test_loss, " avg accuracy: ", accuracy_test_set)

    return average_test_loss, accuracy_test_set


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
    fileName = "..\\..\\data\\stl10_binary\\train_X.bin"
    X_train = read_all_images(fileName)

    train_y_File = "..\\..\\data\\stl10_binary\\train_y.bin"
    y_train = read_labels(train_y_File)

    testFile = "..\\..\\data\\stl10_binary\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "..\\..\\data\\stl10_binary\\test_y.bin"
    targets = read_labels(test_y_File)

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    linearClassifier = SupervisedClassifier(INPUT_NET).to('cuda')
    optimizer = torch.optim.Adam(linearClassifier.parameters(), lr=LEARNING_RATE_DEFAULT)

    filepath = '..\\models\\supervised_encoder' + '.model'
    loss_net_path = os.path.join(script_directory, filepath)

    best_accuracy = 0
    iter_acc = 0
    patience = 50

    for iteration in range(MAX_STEPS_DEFAULT):
        linearClassifier.train()
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        preds, mim = forward_block(X_train, ids, linearClassifier, optimizer, train, y_train)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            linearClassifier.eval()
            print()
            print("==============================================================")
            print("ITERATION: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best acc: ", iter_acc,
                  ": ", best_accuracy)
            print("patience: ", patience)

            print()
            loss, acc = measure_acc_augments(X_test, linearClassifier, targets)

            if best_accuracy < acc:
                best_accuracy = acc
                iter_acc = iteration
                patience = 0
                torch.save(linearClassifier, loss_net_path)
                print("models saved iter: " + str(iteration))
            else:
                patience += 1



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
