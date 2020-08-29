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
INPUT_NET = 2048
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
encoder_name = "..\\binary_brain\\brain_models\\most_clusters_encoder"
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


def forward_block(X, ids, classifier, optimizer, train, targets):
    images = X[ids, :]

    with torch.no_grad():
        # encoding, p1, p2, p3, p4 = encoder(images.to('cuda'))
        encoding, p1, binaries = encoder(images.to('cuda'))

    print(binaries[0])
    print(binaries[1])
    print(binaries[2])
    input()
    #concat = torch.cat([p1, p2, p3, p4], dim=1)

    preds = classifier(binaries)

    # tensor_targets = torch.LongTensor(targets[ids]).unsqueeze(dim=1).cuda()
    # y_onehot = torch.LongTensor(BATCH_SIZE_DEFAULT, 10).cuda()
    # y_onehot.zero_()
    #
    # y_onehot.scatter_(1, tensor_targets, 1)

    targets_tensor = Variable(torch.LongTensor(targets[ids])).to('cuda')

    cross_entropy_loss = loss(preds, targets_tensor)

    if train:

        for p in encoder.parameters():
            p.requires_grad = False

        optimizer.zero_grad()
        cross_entropy_loss.backward()
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

    ### Train data ###
    train_data = sio.loadmat('data\\train_32x32.mat')
    X_train = preproccess(train_data['X'])
    print("x train shape", X_train.shape)

    targets_train = train_data['y'].squeeze(1)
    print("y train", targets_train.shape)
    targets_train = np.array([x % 10 for x in targets_train])

    ### Test data ###
    test_data = sio.loadmat('data\\test_32x32.mat')
    X_test = preproccess(test_data['X'])
    print("x test shape", X_test.shape)

    targets_test = test_data['y'].squeeze(1)
    print("y test", targets_test.shape)
    targets_test = np.array([x % 10 for x in targets_test])

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'encoders\\encoder_' + str(0) + '.model'
    path_to_model = os.path.join(script_directory, filepath)

    linearClassifier = LinearNetSVHN(INPUT_NET).cuda()
    optimizer = torch.optim.Adam(linearClassifier.parameters(), lr=LEARNING_RATE_DEFAULT)

    best_accuracy = 0
    iter_acc = 0
    patience = 50

    for iteration in range(MAX_STEPS_DEFAULT):
        linearClassifier.train()
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        preds, mim = forward_block(X_train, ids, linearClassifier, optimizer, train, targets_train)

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
            for info in DESCRIPTION:
                print(info)
            print()
            loss, acc = measure_acc_augments(X_test, linearClassifier, targets_test)

            if best_accuracy < acc:
                best_accuracy = acc
                iter_acc = iteration
                patience = 0

                print("models saved iter: " + str(iteration))
            else:
                patience += 1

            if patience > PATIENCE:
                print("For ", patience, " iterations we do not have a better accuracy")
                print("best accuracy: ", best_accuracy, " at iter: ", iter_acc)
                print("accuracy at stop: ", acc, "loss at stop: ", loss)

                file = open("experiments.txt", "a")
                for info in DESCRIPTION:
                    file.write(info)

                accuracy_info = ",  BEST ACCURACY: " + str(best_accuracy) + " at iter: " + str(iter_acc) + "\n"
                file.write(accuracy_info)
                file.close()
                break


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
