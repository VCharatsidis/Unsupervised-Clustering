from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import scipy.io as sio
import argparse
import os

from bin_brain_encoder import BinBrain

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

BATCH_SIZE_DEFAULT = 500

#INPUT_NET = 3072
INPUT_NET = 4608
SIZE = 32
SIZE_Y = 20
NETS = 1

DROPOUT = 0
class_n = 2048
CLASSES = [class_n, class_n, class_n, class_n, class_n]
DESCRIPTION = " Image size: "+str(SIZE) + " , Dropout2d: "+str(DROPOUT)+" , Classes: "+str(CLASSES)

EVAL_FREQ_DEFAULT = 1000
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None

best_cluster_accuracies = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

class_numbers = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 0: 0}


def get_targets(number_classes, priors):
    ones = torch.ones([number_classes])
    tensor_prirors = ones * priors
    print(tensor_prirors)
    return tensor_prirors.to('cuda')

global_priors = [0.19, 0.15, 0.11, 0.1, 0.09, 0.08, 0.08, 0.07, 0.06, 0.07]
cuda_tensor_priors = get_targets(10, np.array(global_priors))


transformations_dict ={0: "original",
                       1: "scale",
                       2: "rotate",
                       3: "reverse pixel value",
                       4: "sobel total",
                       5: "sobel x",
                       6: "sobel y",
                       7: "gaussian blur",
                       8: "horizontal flip",
                       9: "random crop",
                       10: "random crop",
                       11: "random crop",
                       12: "random crop"}

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


def transformation(id, image):

    fourth = BATCH_SIZE_DEFAULT//4

    if id == 0:
        return color_jitter(image)
    elif id == 1:
        return scale(color_jitter(image), (image.shape[2] - 8, image.shape[3] - 8), 4, fourth)
    elif id == 2:
        return rotate(color_jitter(image), 30)
    elif id == 3:
        return torch.abs(1 - color_jitter(image))
    elif id == 4:
        return sobel_total(color_jitter(image), fourth)
    elif id == 5:
        return sobel_filter_x(color_jitter(image), fourth)
    elif id == 6:
        return sobel_filter_y(color_jitter(image), fourth)
    elif id == 7:
        return gaussian_blur(color_jitter(image))

    elif id == 8:
        t = random_crop(color_jitter(image), 22, fourth, 18)
        scaled_up = scale_up(t, 32, 20, fourth)
        blured = gaussian_blur(scaled_up)
        return blured

    elif id == 9:
        t = random_crop(color_jitter(image), 22, fourth, 18)
        scaled_up = scale_up(t, 32, 20, fourth)
        sobeled = sobel_total(scaled_up, fourth)
        return sobeled

    elif id == 10:
        t = random_crop(color_jitter(image), 22, fourth, 18)
        scaled_up = scale_up(t, 32, 20, fourth)
        rev = torch.abs(1 - scaled_up)
        return rev

    elif id == 11:
        t = random_crop(color_jitter(image), 22, fourth, 18)
        scaled_up = scale_up(t, 32, 20, fourth)
        rot = rotate(scaled_up, 40)
        return rot

    elif id == 11:
        return horizontal_flip(color_jitter(image), fourth)


    print("Error in transformation of the image.")
    return image


def energy_loss(pred):
    vertical_mean = pred.mean(dim=0)
    reverse_vertical_mean = 1 - vertical_mean
    log_rev_vert_mean = - torch.log(reverse_vertical_mean)
    energy = log_rev_vert_mean.mean(dim=0)

    return energy


def entropy_minmax_loss(preds_1, preds_2):
    energy_loss_1 = energy_loss(preds_1)
    energy_loss_2 = energy_loss(preds_2)

    mean_energy_loss = (energy_loss_1 + energy_loss_2) / 2

    product = preds_1 * preds_2

    # product_horizontal_mean = product.mean(dim=1)
    # product_horizontal_mean[(product_horizontal_mean < EPS).data] = EPS
    # log_horizontal_mean = - torch.log(product_horizontal_mean)
    # batch_horizontal_mean = log_horizontal_mean.mean(dim=0)

    product_vertical_mean = product.mean(dim=0)
    product_vertical_mean[(product_vertical_mean < EPS).data] = EPS
    log_vertical_mean = - torch.log(product_vertical_mean)
    batch_vertical_mean = log_vertical_mean.mean(dim=0)

    sum_mean = 0
    negative_samples = 100
    for i in range(negative_samples):
        #derangement = random_derangement(BATCH_SIZE_DEFAULT)
        a_range = [(x + i + 1) % BATCH_SIZE_DEFAULT for x in range(BATCH_SIZE_DEFAULT)]
        #print(a_range)

        rev_deranged_prod = 1 - product[a_range, :]
        negative = product * rev_deranged_prod
        negative_mean = negative.mean(dim=1)
        negative_mean[(negative_mean < EPS).data] = EPS
        log_neg_mean = torch.log(negative_mean)
        log_mean = - log_neg_mean.mean(dim=0)

        sum_mean += log_mean

    mean_negatives = sum_mean / negative_samples

    total_loss = 0.9 * mean_energy_loss + 0.1 * mean_negatives  # + 0.05 * batch_vertical_mean

    return total_loss


def forward_block(X, ids, encoder, optimizer, train):
    number_transforms = 12
    aug_ids = np.random.choice(number_transforms, size=number_transforms, replace=False)

    image = X[ids, :]

    fourth = BATCH_SIZE_DEFAULT//4
    image_1 = transformation(aug_ids[0], image[0:fourth])
    image_2 = transformation(aug_ids[1], image[0:fourth])

    image_3 = transformation(aug_ids[2], image[fourth: 2*fourth])
    image_4 = transformation(aug_ids[3], image[fourth: 2*fourth])

    image_5 = transformation(aug_ids[4], image[2*fourth: 3*fourth])
    image_6 = transformation(aug_ids[5], image[2*fourth: 3*fourth])

    image_7 = transformation(aug_ids[6], image[3*fourth:])
    image_8 = transformation(aug_ids[7], image[3*fourth:])

    image_1 = torch.cat([image_1, image_3, image_5, image_7], dim=0)
    image_2 = torch.cat([image_2, image_4, image_6, image_8], dim=0)

    _, _, test_preds_1 = encoder(image_1.to('cuda'))
    _, _, test_preds_2 = encoder(image_2.to('cuda'))

    test_total_loss = entropy_minmax_loss(test_preds_1, test_preds_2)

    if train:
        optimizer.zero_grad()
        test_total_loss.backward()
        optimizer.step()

    return test_preds_1, test_preds_2, test_total_loss


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
    #sobel = transformation(4, images)
    _, _, test_preds_1 = encoder(images.to('cuda'))

    return test_preds_1


def measure_acc_augments(X_test, colons, targets):
    def measure_acc_augments(X_test, encoder, targets):
        print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
        size = BATCH_SIZE_DEFAULT
        runs = len(X_test) // size
        avg_loss = 0

        virtual_clusters = {}
        for i in range(CLASSES):
            virtual_clusters[i] = []

        for j in range(runs):
            test_ids = range(j * size, (j + 1) * size)

            images = X_test[test_ids, :]

            _, p = encoder(images.to('cuda'))

            for i in range(p.shape[0]):
                val, index = torch.max(p[i], 0)
                verdict = int(index.data.cpu().numpy())

                label = targets[test_ids[i]]
                if label == 10:
                    label = 0

                print_dict[label].append(verdict)
                virtual_clusters[verdict].append(label)

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

        total_miss_percentage = total_miss / (runs * size)

        print()
        print("AUGMENTS avg loss: ", avg_loss / runs,
              " miss: ", total_miss,
              " data: ", runs * size,
              " miss percent: ", total_miss_percentage)
        print("Clusters found: " + str(len(clusters)) + " " + str(clusters))
        print()

        print(virtual_clusters)
        total_miss_virtual = 0
        for element in virtual_clusters.keys():
            if len(virtual_clusters[element]) == 0:
                continue
            virtual_length = len(virtual_clusters[element])

            virtual_misses = miss_classifications(virtual_clusters[element])
            total_miss_virtual += virtual_misses

            mfe = most_frequent(virtual_clusters[element])

            print("cluster: ",
                  element,
                  ", most frequent: ",
                  labels_to_imags[mfe],
                  ", miss-classifications: ",
                  virtual_misses,
                  ", miss percentage: ",
                  virtual_misses / virtual_length)

        miss_virtual_percentage = total_miss_virtual / (runs * size)
        print("miss virtual percentage: ", miss_virtual_percentage)

        return total_miss_percentage, len(clusters), miss_virtual_percentage


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


def preproccess(x):
    x = to_tensor(x)

    x = x.transpose(0, 2)
    x = x.transpose(1, 3)
    x = x.transpose(0, 1)

    pad = (SIZE-SIZE_Y) // 2
    x = x[:, :, :, pad: SIZE - pad]
    x = rgb2gray(x)

    x = x.unsqueeze(0)
    x = x.transpose(0, 1)
    x /= 255

    return x


def train():
    unsupervised_data = sio.loadmat('..\\SVHN\\data\\extra_32x32.mat')
    train_data = sio.loadmat('..\\SVHN\\data\\train_32x32.mat')
    test_data = sio.loadmat('..\\SVHN\\data\\test_32x32.mat')

    original_train = preproccess(train_data['X'])
    train_targets = train_data['y'].squeeze(1)
    print("y test", train_targets.shape)

    train_targets = np.array([x % 10 for x in train_targets])

    # for target in train_targets:
    #     class_numbers[target] += 1

    X_train = preproccess(unsupervised_data['X'])
    print("x train shape", X_train.shape)

    # access to the dict

    X_test = preproccess(train_data['X'])
    print("x test shape", X_test.shape)

    targets = train_data['y'].squeeze(1)
    print("y test", targets.shape)

    targets = np.array([x % 10 for x in targets])

    # for target in targets:
    #     class_numbers[target] += 1
    #
    # sum = len(targets) + len(train_targets)
    # print("sum ", sum)
    # print(class_numbers)
    #
    # for c in class_numbers:
    #     print(c, " : ", ((class_numbers[c] / sum) * 100))

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'brain_models\\most_clusters_encoder' + '.model'
    clusters_net_path = os.path.join(script_directory, filepath)

    filepath = 'brain_models\\best_loss_encoder' + '.model'
    loss_net_path = os.path.join(script_directory, filepath)

    encoder = BinBrain(1, INPUT_NET, DROPOUT, CLASSES).to('cuda')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    min_miss_percentage = 50
    max_loss_iter = 0
    most_clusters = 0
    most_clusters_iter = 0

    print(encoder)
    print("X_train: ", X_train.shape, " X_test: ", X_test.shape, " targets: ", targets.shape)
    total_mean = torch.ones([CLASSES[0]]) * (1/CLASSES[0])
    total_mean = total_mean.to('cuda')

    test_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    for idx, i in enumerate(targets):
        test_dict[i].append(idx)

    test_best_loss = 1000

    avg_loss = 0
    for iteration in range(MAX_STEPS_DEFAULT):
        encoder.train()

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True

        p1, p2, mim, orig_image, aug_ids = forward_block(X_train, ids, encoder, optimizer, train)
        avg_loss += mim.item()

        test_loss = 0
        if iteration % EVAL_FREQ_DEFAULT == 0:

            print("==================================================================================")
            print("avg train loss : ", avg_loss/BATCH_SIZE_DEFAULT)
            avg_loss = 0
            encoder.eval()

            print("ITERATION: ", iteration,
                  ",  batch size: ", BATCH_SIZE_DEFAULT,
                  ",  lr: ", LEARNING_RATE_DEFAULT,
                  ",  best loss iter: ", max_loss_iter,
                  "-", min_miss_percentage,
                  ",  most clusters iter: ", most_clusters_iter,
                  "-", most_clusters,
                  ",", DESCRIPTION)

            # test_ids = []
            #
            # for i in range(0, 10):
            #     test_ids += random.sample(test_dict[i], samples_per_cluster)

            #test_ids = np.random.choice(len(original_train), size=BATCH_SIZE_DEFAULT, replace=False)
            #p1, p2, mim, orig_image, test_ids = forward_block(original_train, test_ids, encoder, optimizer, False)
            #classes_dict, numbers_classes_dict = print_info(p1, p2, train_targets, test_ids)

            test_loss = measure_acc_augments(X_test, encoder, targets, total_mean)

            if test_loss < test_best_loss:
                test_best_loss = test_loss
                max_loss_iter = iteration
                print("models saved iter: " + str(iteration))
                torch.save(encoder, clusters_net_path)


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

    print("")
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