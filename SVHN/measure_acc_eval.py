from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from stl_utils import *
import scipy.io as sio
from torchvision.utils import make_grid
import matplotlib
from linear_layer_svhn import LinearNetSVHN
from train_svhn_2 import preproccess
import sys

EPS = sys.float_info.epsilon
# Default constants

LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 300
INPUT_NET = 32
SIZE = 32
NETS = 1
EVAL_FREQ_DEFAULT = 100
PATIENCE = 100

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
FLAGS = None


############ UNSUPERVISED INFO #############################
classes_encoder = 30

conv_layers_encoder = 4
number_filters = 512
linear_layers_encoder = 1

batch_size = 300
lr = 1e-4

augments_compared = 6
heads = 1

# encoder_name = "push_best_loss"
# encoder_name = "push_most_clusters"
encoder_name = "svhn_models\\clusters12_120k_0.23miss_9aug"
encoder_name2 = "clusters12_120k_0.23miss_9aug"
#encoder_name = "2aug_20_classes_0.57miss"

# encoder_name = "best_loss_encoder"
SUPERVISED_FILE_NAME = "supervised_encoder_loss"
# # SUPERVISED_FILE_NAME = "supervised_encoder_acc"
# encoder_name = SUPERVISED_FILE_NAME

#encoder = torch.load("svhn_models\\"+encoder_name+".model")
encoder = torch.load(encoder_name+".model")
encoder.eval()

# encoder2 = torch.load("svhn_models\\"+encoder_name2+".model")
# encoder2.eval()

labels_to_imags = {1: "1",
                   2: "2",
                   3: "3",
                   4: "4",
                   5: "5",
                   6: "6",
                   7: "7",
                   8: "8",
                   9: "9",
                   0: "0"}

images_to_labels = {"1": 1,
                    "2": 2,
                    "3": 3,
                    "4": 4,
                    "5": 5,
                    "6": 6,
                    "7": 7,
                    "8": 8,
                    "9": 9,
                    "0": 0}


DESCRIPTION = ["Supervised NET with 75% augments per batch."]

DESCRIPTION = ["LOSS: product loss multiplied by mean ", " Image size: " + str(SIZE)\
              +",  BATCH SIZE: " + str(batch_size)\
              +",  lr: " + str(lr) + ",  train iters: 300000"
              ,",  Classes: " + str(classes_encoder)\
              ,",  embedding dim: " + str(INPUT_NET)\
              +",  conv layers: " + str(conv_layers_encoder)\
              +",  linear layers: " + str(linear_layers_encoder)\
              ,",  number filters: " + str(number_filters)\
              ,",  augments compared: " + str(augments_compared)\
              +",  heads: " + str(heads)\
              +",  Policy: " + "  " + encoder_name]

loss = nn.CrossEntropyLoss()


def accuracy(predictions, targets):
   predictions = predictions.cpu().detach().numpy()
   preds = np.argmax(predictions, 1)
   result = preds == targets
   sum = np.sum(result)

   return sum


def measure_acc_augments(X_test, classifier, targets):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    runs = len(X_test) // BATCH_SIZE_DEFAULT

    sum_correct = 0
    size = BATCH_SIZE_DEFAULT

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        test_ids = np.array(test_ids)
        optimizer = []

        images = X_test[test_ids]
        with torch.no_grad():
            # encoding, p1, p2, p3, p4 = encoder(images.to('cuda'))
            encoding, p1, preds = classifier(images.to('cuda'))

        for i in range(preds.shape[0]):
            _, mean_index = torch.max(preds[i], 0)
            verdict = int(mean_index.data.cpu().numpy())
            label = targets[test_ids[i]]

            if label == 10:
                label = 0

            print_dict[label].append(verdict)


    total_miss = 0
    clusters = set()
    for element in print_dict.keys():
        length = len(print_dict[element])
        misses = miss_classifications(print_dict[element])
        total_miss += misses

        mfe = most_frequent(print_dict[element])
        clusters.add(mfe)
        print("cluster: ",
              labels_to_imags[element],
              ", most frequent: ",
              mfe,
              ", miss-classifications: ",
              misses,
              ", miss percentage: ",
              misses / length)

    miss_percentage = total_miss / (runs * size)
    print()
    print("AUGMENTS avg loss: ",
          " miss: ", total_miss,
          " data: ", runs * size,
          " miss percent: ", miss_percentage)
    print("Clusters found: " + str(len(clusters)) + " " + str(clusters))
    print()


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

    ### Train data ###
    # train_data = sio.loadmat('data\\train_32x32.mat')
    # X_train = preproccess(train_data['X'])
    # print("x train shape", X_train.shape)
    #
    # targets_train = train_data['y'].squeeze(1)
    # print("y train", targets_train.shape)
    # targets_train = np.array([x % 10 for x in targets_train])

    ### Test data ###
    test_data = sio.loadmat('data\\test_32x32.mat')
    X_test = preproccess(test_data['X'])
    print("x test shape", X_test.shape)

    targets_test = test_data['y'].squeeze(1)
    print("y test", targets_test.shape)
    targets_test = np.array([x % 10 for x in targets_test])

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    measure_acc_augments(X_test, encoder, targets_test)


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
