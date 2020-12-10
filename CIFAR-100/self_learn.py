from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.utils.prune as prune

import sys

import argparse
import os

from SelfNet import SelfNet

from image_utils import *
import random

from torchvision.utils import make_grid
import matplotlib
import pickle

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

EPS = sys.float_info.epsilon

#EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4

MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 128
TR = 1
QUEUE = 300

EMBEDINGS = 100
SIZE = 32
SIZE_Y = 32
NETS = 1

EPOCHS = 1000

CLASSES = 100
DESCRIPTION = " Image size: " + str(SIZE) + " , Classes: " + str(CLASSES)

EVAL_FREQ_DEFAULT = 250
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None

square = torch.ones(BATCH_SIZE_DEFAULT, BATCH_SIZE_DEFAULT)
adj_matrix = square.fill_diagonal_(0).cuda()


first = True


class_numbers = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 0: 0}


transformations_dict = {0: "original 0",
                       1: "scale 1",
                       2: "rotate 2",
                       3: "reverse pixel value 3",
                       4: "sobel total 4",
                       5: "sobel x 5",
                       6: "sobel y 6",
                       7: "gaussian blur 7",
                       8: "randcom_crop_upscale_gauss_blur 8",
                       9: "randcom_crop_upscale sobel 9",
                       10: "random crop reverse pixel 10",
                       11: "random crop rotate 11",
                       12: "random crop soble rotate 12",
                       13: "randcom_crop_upscale 13",
                        14: "randcom_crop_upscale 14",
                        15:"randcom_crop_upscale sobel y 15",
                        16: " randcom crop 18x18 16"}


labels_to_imags = {}
for i in range(CLASSES):
    labels_to_imags[i] = i


def save_images(images, transformation):
    print(transformations_dict[transformation])
    numpy_cluster = images.cpu().detach()
    save_cluster(numpy_cluster, transformations_dict[transformation], 0)


def new_agreement(product, denominator, rev_prod):
    attraction = (torch.mm(product, product.transpose(0, 1)) + EPS) * (1 - adj_matrix)
    repel = (torch.mm(product, rev_prod.transpose(0, 1)) + EPS) * adj_matrix

    repel = repel / denominator

    total_matrix = - torch.log(repel + attraction)

    mean_total = total_matrix.mean()

    return mean_total

# def new_agreement(product, denominator, rev_prod):
#     attraction = (torch.mm(product, product.transpose(0, 1)) + EPS) * (1 - adj_matrix)
#     repel = (torch.mm(product, rev_prod.transpose(0, 1)) + EPS) * adj_matrix
#
#     attraction = attraction / denominator
#     repel = repel / denominator
#
#     total_matrix = - torch.log(repel + attraction)
#
#     attraction_bonus = 0.5 * (BATCH_SIZE_DEFAULT - 1) * (1 - adj_matrix) * total_matrix
#
#     total_matrix = total_matrix + attraction_bonus.fill_diagonal_(0)
#
#     mean_total = total_matrix.mean()
#
#     return mean_total


def queue_agreement(product, denominator, rev_prod):
    transposed = rev_prod.transpose(0, 1)

    nondiag = torch.mm(product, transposed) + EPS
    nondiag = nondiag / denominator

    log_nondiag = - torch.log(nondiag)
    negative = log_nondiag.mean()

    return negative


def forward_block(X, ids, encoder, optimizer, train, rev_product):
    global first

    image = X[ids, :]

    _, _, all_predictions, = encoder(image.to('cuda'))

    denominator = all_predictions.sum(dim=1)
    denominator = denominator.unsqueeze(dim=1) + 1
    current_reverse = 1 - all_predictions

    new_loss = new_agreement(all_predictions, denominator, current_reverse)

    if first or not train:
        total_loss = new_loss
        rev_product = current_reverse.detach()
        first = False

    else:
        old_loss = queue_agreement(all_predictions, denominator, rev_product)
        rev_product = torch.cat([rev_product, current_reverse.detach()])
        total_loss = new_loss + old_loss

    if train:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return all_predictions, all_predictions, total_loss, rev_product


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def measure_acc_augments(x_test, encoder, rev_product):
    size = BATCH_SIZE_DEFAULT
    runs = len(x_test) // size
    sum_loss = 0

    print(rev_product.shape)

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)

        with torch.no_grad():
            test_preds_1, test_preds_2, test_total_loss, rev_product = forward_block(x_test, test_ids, encoder, [], False, rev_product)

        sum_loss += test_total_loss.item()

    avg_loss = sum_loss / runs

    print()
    print("Avg test loss: ", avg_loss)
    print()

    return avg_loss


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


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def train():
    global first
    with open('data\\train', 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')

    meta = unpickle('data\\meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]

    train = unpickle('data\\train')

    filenames = [t.decode('utf8') for t in train[b'filenames']]
    train_fine_labels = train[b'fine_labels']
    train_data = train[b'data']

    test = unpickle('data\\test')

    filenames = [t.decode('utf8') for t in test[b'filenames']]
    targets = test[b'fine_labels']
    test_data = test[b'data']

    X_train = list()
    for d in train_data:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
        image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
        image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
        X_train.append(image)

    X_train = np.array(X_train)
    X_train = preproccess_cifar(X_train)

    print("train shape", X_train.shape)

    X_test = list()
    for d in test_data:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
        image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
        image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
        X_test.append(image)

    X_test = np.array(X_test)
    X_test = preproccess_cifar(X_test)
    print("test shape", X_test.shape)
    targets = np.array(targets)
    print("targets shape", targets.shape)

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'cifar100_models\\self_learn'
    clusters_net_path = os.path.join(script_directory, filepath)

    encoder = SelfNet(3, EMBEDINGS).to('cuda')

    print(encoder)

    #print(list(encoder.brain[0].weight))
    #prune.random_unstructured(encoder.brain[0], name="weight", amount=0.6)
    #print(list(encoder.brain[0].weight))

    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    min_miss_percentage = 100
    max_loss_iter = 0

    test_best_loss = 1000

    print("X_train: ", X_train.shape, " X_test: ", X_test.shape, " targets: ", targets.shape)

    # test_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    # for idx, i in enumerate(targets):
    #     test_dict[i].append(idx)

    avg_loss = 0
    total_iters = 0

    for epoch in range(EPOCHS):
        print("Epoch: ", epoch)
        ids = np.random.choice(len(X_train), size=len(X_train), replace=False)

        runs = len(X_train) // BATCH_SIZE_DEFAULT

        rev_product = torch.ones([BATCH_SIZE_DEFAULT, EMBEDINGS]).cuda()
        first = True
        iteration = 0

        for j in range(runs):
            current_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
            encoder.train()
            iter_ids = ids[current_ids]

            train = True
            probs10, probs10_b, total_loss, rev_product = forward_block(X_train, iter_ids, encoder, optimizer, train, rev_product)
            avg_loss += total_loss.item()
            iteration += 1
            total_iters += 1

            if iteration >= QUEUE:
                rev_product = rev_product[TR * BATCH_SIZE_DEFAULT:, :]

        print("==================================================================================")
        print("batch mean ones: ",
              (np.where(probs10.data.cpu().numpy() > 0.5))[0].shape[0] / (probs10[0].shape[0] * BATCH_SIZE_DEFAULT))

        count_common_elements(probs10)
        print()

        print("train avg loss : ", avg_loss / runs)
        avg_loss = 0
        encoder.eval()

        test_loss = measure_acc_augments(X_test, encoder, rev_product)

        if test_loss < test_best_loss:
            test_best_loss = test_loss
            max_loss_iter = total_iters
            min_miss_percentage = test_loss
            print("models saved iter: " + str(total_iters))
            torch.save(encoder, clusters_net_path + "_" + str(epoch//100) + '.model')

        print("EPOCH: ", epoch,
              "Total ITERATION: ", total_iters,
              " epoch iter: ", iteration,
              ",  batch size: ", BATCH_SIZE_DEFAULT,
              ",  lr: ", LEARNING_RATE_DEFAULT,
              ",  best loss iter: ", max_loss_iter,
              "-", min_miss_percentage
              , DESCRIPTION)


def count_common_elements(p):
    sum_commons = 0
    counter = 0
    for i in range(p.shape[0]):
        for j in range(p.shape[0]):
            if i == j:
                continue

            product = p[i].data.cpu().numpy() * p[j].data.cpu().numpy()
            commons = np.where(product > 0.5)[0].shape[0]
            #print(commons)

            sum_commons += commons
            counter += 1

    print("Mean common elements: ", (sum_commons / EMBEDINGS) / counter)

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