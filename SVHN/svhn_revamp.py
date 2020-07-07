from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import scipy.io as sio
import argparse
import os

from svhn_encoder import SVHNencoderNet
from stl_utils import *
import random

from torchvision.utils import make_grid
import matplotlib
from torchvision import models
EPS = sys.float_info.epsilon


fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

#EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4
# ANCHOR = 1e-4
# SPIKE = 3e-3
MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 1200

#INPUT_NET = 3072
INPUT_NET = 4608
SIZE = 32
SIZE_Y = 20
NETS = 1

DROPOUT = 0
class_n = 12
CLASSES = [class_n, class_n, class_n, class_n, class_n]
DESCRIPTION = " Image size: "+str(SIZE) + " , Dropout2d: "+str(DROPOUT)+" , Classes: "+str(CLASSES)

EVAL_FREQ_DEFAULT = 20
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None


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

images_to_labels ={"1": 1,
                   "2": 2,
                   "3": 3,
                   "4": 4,
                   "5": 5,
                   "6": 6,
                   "7": 7,
                   "8": 8,
                   "9": 9,
                   "0": 0}


def get_targets(number_classes):
    return (torch.ones([number_classes]).to('cuda')) / number_classes


def entropy_minmax_loss(preds_1, preds_2):
    product = preds_1 * preds_2
    product = product.mean(dim=0)
    product[(product < EPS).data] = EPS
    total_loss = - torch.log(product).mean(dim=0)

    return total_loss


def get_final_target(encoder, transformations):
    targets = []
    with torch.no_grad():
        # sum = torch.zeros([BATCH_SIZE_DEFAULT, CLASSES[0]]).to('cuda')
        # for idx, t in enumerate(transformations):
        #     _, preds = encoder(t.detach().to('cuda'))
        #     targets.append(preds.detach())
        #     sum += preds
        #
        # final_target = sum / len(transformations)

        final_target = torch.ones([BATCH_SIZE_DEFAULT, CLASSES[0]]).to('cuda')
        for idx, t in enumerate(transformations):
            _, preds = encoder(t.detach().to('cuda'))
            targets.append(preds.detach())
            final_target *= preds

    return targets, final_target


def calc_targets(encoder, transformations):
    with torch.no_grad():
        _, targets = encoder(transformations.detach().to('cuda'))

    return targets


def retarget(targets):
    with torch.no_grad():
        # sum = torch.zeros([BATCH_SIZE_DEFAULT, CLASSES[0]]).to('cuda')
        #
        # for idx, t in enumerate(targets):
        #     sum += t
        #
        # final_target = sum / len(targets)

        final_target = torch.ones([BATCH_SIZE_DEFAULT, CLASSES[0]]).to('cuda')
        for t in targets:
            final_target *= t

    return final_target


def big_forward(X, ids, encoder, optimizer):
    images = X[ids, :]

    transforms = []
    transforms.append(color_jitter(images))
    transforms.append(scale(color_jitter(images), (images.shape[2] - 8, images.shape[3] - 8), 4, BATCH_SIZE_DEFAULT))
    transforms.append(rotate(color_jitter(images), 30))
    transforms.append(torch.abs(1 - color_jitter(images)))
    transforms.append(sobel_total(color_jitter(images), BATCH_SIZE_DEFAULT))
    transforms.append(sobel_filter_x(color_jitter(images), BATCH_SIZE_DEFAULT))
    transforms.append(sobel_filter_y(color_jitter(images), BATCH_SIZE_DEFAULT))

    transforms.append(horizontal_flip(color_jitter(images), BATCH_SIZE_DEFAULT))
    transforms.append(gaussian_blur(color_jitter(images)))

    t = random_crop(color_jitter(images), 18, BATCH_SIZE_DEFAULT, 18)
    transforms.append(scale_up(t, 32, 20, BATCH_SIZE_DEFAULT))

    number_transformations = len(transforms)

    tfs_ids = np.random.choice(number_transformations, size=number_transformations, replace=False)
    targets, final_target = get_final_target(encoder, transforms)
    #target = calc_targets(encoder, transforms[tfs_ids[0]])

    sum_loss = 0

    times = 1
    for j in range(times):

        for i in range(1, number_transformations):
            next_transformation = tfs_ids[i]
            prev_target, targets[next_transformation], mim = forward_block(encoder, optimizer, final_target.detach(), transforms[next_transformation])
            final_target = retarget(targets)
            sum_loss += mim.item()

        tfs_ids = np.random.choice(number_transformations, size=number_transformations, replace=False)
        final_target = retarget(targets)
        #target = calc_targets(encoder, transforms[tfs_ids[0]])

    loss = sum_loss / (number_transformations * times)

    return prev_target, targets[0], loss, images


def forward_block(encoder, optimizer, targets, transformations):
    _, predictions = encoder(transformations.to('cuda'))

    test_total_loss = entropy_minmax_loss(targets.detach(), predictions)

    optimizer.zero_grad()
    test_total_loss.backward()
    optimizer.step()

    return targets, predictions, test_total_loss


def batch_entropy(pred, targets):
    batch_mean_preds = pred.mean(dim=0)
    H_batch = torch.log(batch_mean_preds).mean()

    return H_batch


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"new_images/iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def accuracy_block(X, ids, encoder):
    images = X[ids, :]
    _, test_preds_1 = encoder(images.to('cuda'))

    return test_preds_1


def measure_acc_augments(X_test, colons, targets):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    size = BATCH_SIZE_DEFAULT
    runs = len(X_test)//size
    avg_loss = 0

    print()

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)
        p = accuracy_block(X_test, test_ids, colons)

        for i in range(p.shape[0]):
            _, mean_index = torch.max(p[i], 0)
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

    print()
    print("AUGMENTS avg loss: ", avg_loss / runs,
          " miss: ", total_miss,
          " data: ", runs * size,
          " miss percent: ", total_miss / (runs * size))
    print("Clusters found: " + str(len(clusters)) + " " + str(clusters))
    print()

    return avg_loss/runs, len(clusters)


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


def preproccess(X_train):
    X_train = to_tensor(X_train)

    X_train = X_train.transpose(0, 2)
    X_train = X_train.transpose(1, 3)
    X_train = X_train.transpose(0, 1)

    pad = (SIZE-SIZE_Y) // 2
    X_train = X_train[:, :, :, pad: SIZE-pad]
    X_train = rgb2gray(X_train)

    X_train = X_train.unsqueeze(0)
    X_train = X_train.transpose(0, 1)
    X_train /= 255

    return X_train


def train():
    unsupervised_data = sio.loadmat('data\\extra_32x32.mat')
    train_data = sio.loadmat('data\\train_32x32.mat')

    X_train = preproccess(unsupervised_data['X'])
    print("x train shape", X_train.shape)

    # access to the dict

    X_test = preproccess(train_data['X'])
    print("x test shape", X_test.shape)

    targets = train_data['y'].squeeze(1)
    print("y test", targets.shape)

    targets = np.array([x % 10 for x in targets])

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'svhn_models\\most_clusters_encoder' + '.model'
    clusters_net_path = os.path.join(script_directory, filepath)

    filepath = 'svhn_models\\best_loss_encoder' + '.model'
    loss_net_path = os.path.join(script_directory, filepath)

    encoder = SVHNencoderNet(1, INPUT_NET, DROPOUT, CLASSES).to('cuda')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    max_loss = 1999
    max_loss_iter = 0
    most_clusters = 0
    most_clusters_iter = 0

    print(encoder)
    print("X_train: ", X_train.shape, " X_test: ", X_test.shape, " targets: ", targets.shape)

    test_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    for idx, i in enumerate(targets):
        test_dict[i].append(idx)

    for iteration in range(MAX_STEPS_DEFAULT):
        encoder.train()
        samples_per_cluster = BATCH_SIZE_DEFAULT // 10

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)

        p1, p2, mim, orig_image = big_forward(X_train, ids, encoder, optimizer)

        if iteration % EVAL_FREQ_DEFAULT == 0:

            encoder.eval()
            print("==================================================================================")
            print("ITERATION: ", iteration,
                  ",  batch size: ", BATCH_SIZE_DEFAULT,
                  ",  lr: ", LEARNING_RATE_DEFAULT,
                  ",  best loss iter: ", max_loss_iter,
                  "-", max_loss,
                  ",  most clusters iter: ", most_clusters_iter,
                  "-", most_clusters,
                  ",", DESCRIPTION)

            test_ids = []

            for i in range(0, 10):
                test_ids += random.sample(test_dict[i], samples_per_cluster)

            p1, p2, mim, orig_image = big_forward(X_test, test_ids, encoder, optimizer)
            classes_dict, numbers_classes_dict = print_info(p1, p2, targets, test_ids)

            loss, clusters = measure_acc_augments(X_test, encoder, targets)

            if clusters >= MIN_CLUSTERS_TO_SAVE and max_loss > loss:
                for key in numbers_classes_dict.keys():
                    numpy_cluster = torch.zeros([len(numbers_classes_dict[key]), 1, SIZE, SIZE_Y])
                    counter = 0
                    for index in numbers_classes_dict[key]:
                        numpy_cluster[counter] = orig_image.cpu().detach()[index]
                        counter += 1
                    if counter > 0:
                        save_cluster(numpy_cluster, key, iteration)

            if clusters >= most_clusters:
                most_clusters = clusters
                most_clusters_iter = iteration
                print("models saved iter: " + str(iteration))
                torch.save(encoder, clusters_net_path)

            if max_loss > loss:
                max_loss = loss
                max_loss_iter = iteration

                print("models saved iter: " + str(iteration))
                torch.save(encoder, loss_net_path)


def to_tensor(X):
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


def print_info(p1, p2, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}
    #image_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    image_dict = {x: [] for x in range(CLASSES[0])}
    numbers_classes_dict = {x: [] for x in range(CLASSES[0])}

    for i in range(len(test_ids)):
        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)

        mean = (p1[i] + p2[i])/2
        _, mean_index = torch.max(mean, 0)
        verdict = int(mean_index.data.cpu().numpy())

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + ", "

        label = targets[test_ids[i]]
        if label == 10:
            label = 0

        numbers_classes_dict[verdict].append(i)
        image_dict[verdict].append(labels_to_imags[label])
        print_dict[label] += string

    for i in print_dict.keys():
        print(labels_to_imags[i], " : ", print_dict[i])

    for i in image_dict.keys():
        print(i, " : ", image_dict[i])

    return image_dict, numbers_classes_dict


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