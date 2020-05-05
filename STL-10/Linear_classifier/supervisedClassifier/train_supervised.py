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

BATCH_SIZE_DEFAULT = 150

INPUT_NET = 4608
SIZE = 32
NETS = 1
EVAL_FREQ_DEFAULT = 200
PATIENCE = 20

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
FLAGS = None
criterion = nn.CrossEntropyLoss().cuda()


def forward_block(X, ids, classifier, optimizer, train, targets, size):
    x_tens = X[ids, :]
    x_train = rgb2gray(x_tens)

    x_tensor = to_tensor(x_train)
    x_tensor = x_tensor.unsqueeze(0)
    images = x_tensor.transpose(0, 1)
    images = images.transpose(2, 3)

    images /= 255

    pad = (96 - SIZE) // 2
    original_image = scale(images, SIZE, pad, size)
    original_image = original_image[:, :, pad:96 - pad, pad:96 - pad]

    if train:
        crop_size = 56
        crop_pad = (96 - crop_size) // 2
        crop_preparation = scale(images, crop_size, crop_pad, size)
        crop_preparation = crop_preparation[:, :, crop_pad:96 - crop_pad, crop_pad:96 - crop_pad]

        crop_size2 = 70
        crop_pad2 = (96 - crop_size2) // 2
        crop_preparation2 = scale(images, crop_size2, crop_pad2, size)
        crop_preparation2 = crop_preparation2[:, :, crop_pad2:96 - crop_pad2, crop_pad2:96 - crop_pad2]
        crop_prep_horizontal2 = horizontal_flip(crop_preparation2)

        horiz_f = horizontal_flip(images, size)

        soft_bin_hf = binary(horiz_f)
        soft_bin_hf = scale(soft_bin_hf, SIZE, pad, size)
        soft_bin_hf = soft_bin_hf[:, :, pad:96 - pad, pad:96 - pad]
        rev_soft_bin_hf = torch.abs(1 - soft_bin_hf)

        original_hfliped = horizontal_flip(original_image, size)

        augments = {0: color_jitter(original_hfliped),
                    1: scale(original_hfliped, SIZE - 8, 4, size),
                    2: random_erease(color_jitter(original_hfliped), size),
                    3: sobel_filter_x(original_hfliped, size),
                    4: sobel_filter_y(original_hfliped, size),
                    5: sobel_total(original_hfliped, size),
                    6: soft_bin_hf,
                    7: rev_soft_bin_hf,
                    8: torch.abs(1 - original_hfliped),
                    9: rotate(original_hfliped, 40),
                    10: scale(original_hfliped, SIZE - 12, 6, size),
                    }

        aug_ids = np.random.choice(len(augments), size=len(augments.keys()), replace=False)

        image_1 = color_jitter(random_crop(crop_preparation, SIZE, size))
        image_2 = color_jitter(random_crop(crop_prep_horizontal2, SIZE, size))
        image_3 = augments[aug_ids[2]]
        image_4 = original_image

        data = torch.cat([image_1.to('cuda'), image_2.to('cuda'), image_3.to('cuda'), image_4.to('cuda')], dim=0)

        _, preds = classifier(data)

        t = [x % 10 for x in targets[ids]]

        tensor_targets = torch.LongTensor(t).to('cuda')

        all_targets = torch.cat([tensor_targets, tensor_targets, tensor_targets, tensor_targets], dim=0)

        cross_entropy_loss = criterion(preds, all_targets)

        optimizer.zero_grad()
        cross_entropy_loss.backward(retain_graph=True)
        optimizer.step()

    else:
        _, preds = classifier(original_image.to('cuda'))

        t = [x % 10 for x in targets[ids]]

        tensor_targets = torch.LongTensor(t).to('cuda')
        print(preds)
        print(tensor_targets)
        input()
        cross_entropy_loss = criterion(preds, tensor_targets)


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
    size = 100
    runs = len(X_test) // size
    avg_loss = 0
    sum_correct = 0

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)
        test_ids = np.array(test_ids)
        optimizer = []
        preds, mim = forward_block(X_test, test_ids, classifier, optimizer, False, targets, size)

        t = [x % 10 for x in targets[test_ids]]

        sum_correct += accuracy(preds, t)
        avg_loss += mim.item()

    average_test_loss = avg_loss / runs
    accuracy_test_set = sum_correct / (runs * size)
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

    #######################################################

    testFile = "..\\..\\data\\stl10_binary\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "..\\..\\data\\stl10_binary\\test_y.bin"
    targets = read_labels(test_y_File)

    #######################################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    supervisedClassifier = SupervisedClassifier(INPUT_NET).to('cuda')
    optimizer = torch.optim.Adam(supervisedClassifier.parameters(), lr=LEARNING_RATE_DEFAULT)

    filepath = '..\\models\\supervised_encoder_loss' + '.model'
    loss_net_path = os.path.join(script_directory, filepath)

    filepath2 = '..\\models\\supervised_encoder_acc' + '.model'
    acc_net_path = os.path.join(script_directory, filepath2)

    best_accuracy = 0
    iter_acc = 0
    patience = 50
    max_loss = 1000

    for iteration in range(MAX_STEPS_DEFAULT):
        supervisedClassifier.train()
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        preds, mim = forward_block(X_train, ids, supervisedClassifier, optimizer, train, y_train, BATCH_SIZE_DEFAULT)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            supervisedClassifier.eval()
            print()
            print("==============================================================")
            print("ITERATION: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best acc: ", iter_acc,
                  ": ", best_accuracy)
            print("patience: ", patience)

            print()
            loss, acc = measure_acc_augments(X_test, supervisedClassifier, targets)

            if max_loss > loss:
                max_loss = loss
                print("models saved iter loss: " + str(iteration))
                torch.save(supervisedClassifier, loss_net_path)

            if best_accuracy < acc:
                best_accuracy = acc
                iter_acc = iteration
                patience = 0
                torch.save(supervisedClassifier, acc_net_path)
                print("models saved iter accuracy: " + str(iteration))
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
