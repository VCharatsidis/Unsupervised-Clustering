from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from stl10_input import read_all_images, read_labels
from UnsupervisedEncoder import UnsupervisedNet
from stl_utils import *
import random
import sys
from torchvision.utils import make_grid
import matplotlib
from torchvision import models

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()


#EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000

BATCH_SIZE_DEFAULT = 300

INPUT_NET = 2048
SIZE = 32
NETS = 1
UNSUPERVISED = True

CLASSES = 12
DESCRIPTION = " Image size: "+str(SIZE) + " , Classes: "+str(CLASSES)

EVAL_FREQ_DEFAULT = 500
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None

cluster_accuracies = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

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


def entropy_minmax_loss(preds_1, preds_2):
    product = preds_1 * preds_2
    product = product.mean(dim=0)
    log_product = torch.log(product)
    total_loss = - log_product.mean(dim=0)

    return total_loss


def forward_block(X, ids, encoder, optimizer, train):
    images = X[ids]

    if train:
        images = rgb2gray(images)

        images = to_tensor(images)
        images = images.unsqueeze(0)
        images = images.transpose(0, 1)
        images = images.transpose(2, 3)

        images /= 255

    number_augments = 12
    aug_ids = np.random.choice(number_augments, size=2, replace=False)

    image_1 = transformation(aug_ids[0], images)
    image_2 = transformation(aug_ids[1], images)

    # show_gray(image_1)
    # show_gray(image_2)

    _, test_preds_1 = encoder(image_1.to('cuda'))
    _, test_preds_2 = encoder(image_2.to('cuda'))

    total_loss = entropy_minmax_loss(test_preds_1, test_preds_2)

    if train:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return test_preds_1, test_preds_2, total_loss


def batch_entropy(pred, targets):
    batch_mean_preds = pred.mean(dim=0)
    H_batch = torch.log(batch_mean_preds).mean()

    return H_batch


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"new_images/iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def transformation(id, image):
    pad = (96 - SIZE) // 2
    fourth = BATCH_SIZE_DEFAULT

    if random.uniform(0, 1) > 0.5:
        image = horizontal_flip(image, fourth)

    if id == 0:
        image = scale(image, SIZE, pad, fourth)
        image = image[:, :, pad:96 - pad, pad:96 - pad]
        color_jit_image = color_jitter(image)
        return color_jit_image

    elif id == 1:
        image = scale(image, SIZE, pad, fourth)
        image = image[:, :, pad:96 - pad, pad:96 - pad]
        color_jit_image = color_jitter(image)
        return scale(color_jit_image, (color_jit_image.shape[2] - 8, color_jit_image.shape[3] - 8), 4,
                     fourth)
    elif id == 2:
        image = scale(image, SIZE, pad, fourth)
        image = image[:, :, pad:96 - pad, pad:96 - pad]
        color_jit_image = color_jitter(image)
        return rotate(color_jit_image, 30)

    elif id == 3:
        image = scale(image, SIZE, pad, fourth)
        image = image[:, :, pad:96 - pad, pad:96 - pad]
        color_jit_image = color_jitter(image)
        return torch.abs(1 - color_jit_image)

    elif id == 4:
        image = scale(image, SIZE, pad, fourth)
        image = image[:, :, pad:96 - pad, pad:96 - pad]
        color_jit_image = color_jitter(image)
        return sobel_total(color_jit_image, fourth)

    elif id == 5:
        image = scale(image, SIZE, pad, fourth)
        image = image[:, :, pad:96 - pad, pad:96 - pad]
        color_jit_image = color_jitter(image)
        return sobel_filter_x(color_jit_image, fourth)

    elif id == 6:
        image = scale(image, SIZE, pad, fourth)
        image = image[:, :, pad:96 - pad, pad:96 - pad]
        color_jit_image = color_jitter(image)
        return sobel_filter_y(color_jit_image, fourth)

    elif id == 7:
        image = scale(image, SIZE, pad, fourth)
        image = image[:, :, pad:96 - pad, pad:96 - pad]
        color_jit_image = color_jitter(image)
        return gaussian_blur(color_jit_image)

    elif id == 8:
        crop_size = 66
        crop_pad = (96 - crop_size) // 2
        color_jit_image = color_jitter(image)
        crop_preparation = scale(color_jit_image, crop_size, crop_pad, fourth)
        crop_preparation = crop_preparation[:, :, crop_pad:96 - crop_pad, crop_pad:96 - crop_pad]
        return random_crop(crop_preparation, SIZE, fourth, SIZE)

    elif id == 9:
        crop_size2 = 76
        crop_pad2 = (96 - crop_size2) // 2
        color_jit_image = color_jitter(image)
        crop_preparation2 = scale(color_jit_image, crop_size2, crop_pad2, fourth)
        crop_preparation2 = crop_preparation2[:, :, crop_pad2:96 - crop_pad2, crop_pad2:96 - crop_pad2]
        return random_crop(crop_preparation2, SIZE, fourth, SIZE)

    elif id == 10:
        image = scale(image, SIZE, pad, fourth)
        image = image[:, :, pad:96 - pad, pad:96 - pad]
        color_jit_image = color_jitter(image)
        if random.uniform(0, 1) > 0.5:
            color_jit_image = gaussian_blur(color_jit_image)
        return random_erease(color_jit_image, fourth)

    elif id == 11:
        crop_size2 = 76
        crop_pad2 = (96 - crop_size2) // 2
        color_jit_image = color_jitter(image)
        crop_preparation2 = scale(color_jit_image, crop_size2, crop_pad2, fourth)
        crop_preparation2 = crop_preparation2[:, :, crop_pad2:96 - crop_pad2, crop_pad2:96 - crop_pad2]
        return random_crop(crop_preparation2, SIZE, fourth, SIZE)

    print("Error in transformation of the image.")
    print("id ", id)
    input()
    return image


def measure_acc_augments(X_test, encoder, targets):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    size = BATCH_SIZE_DEFAULT
    runs = len(X_test)//size
    avg_loss = 0

    virtual_clusters = {}
    for i in range(CLASSES):
        virtual_clusters[i] = []

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)

        images = X_test[test_ids, :]

        pad = (96 - SIZE) // 2
        images = scale(images, SIZE, pad, BATCH_SIZE_DEFAULT)
        images = images[:, :, pad:96 - pad, pad:96 - pad]

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


def train():
    train_path = "..\\data_2\\stl10_binary\\unlabeled_X.bin"
    print(train_path)
    X_train = read_all_images(train_path)

    ########### test ##############################
    testFile = "..\\data\\stl10_binary\\train_X.bin"
    X_validation = read_all_images(testFile)

    X_validation = rgb2gray(X_validation)

    X_validation = to_tensor(X_validation)
    X_validation = X_validation.unsqueeze(0)
    X_validation = X_validation.transpose(0, 1)
    X_validation = X_validation.transpose(2, 3)

    X_validation /= 255

    test_y_File = "..\\data\\stl10_binary\\train_y.bin"
    targets = read_labels(test_y_File)

    targets = np.array([x % 10 for x in targets])

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'new_models\\actual_best' + '.model'
    actual_best_path = os.path.join(script_directory, filepath)

    filepath = 'new_models\\virtual_best' + '.model'
    virtual_best_path = os.path.join(script_directory, filepath)

    encoder = UnsupervisedNet(1, CLASSES).to('cuda')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    min_miss_percentage = 100
    max_loss_iter = 0
    most_clusters = 0
    most_clusters_iter = 0
    min_virtual_miss_percentage = 1000
    best_virtual_iter = 0

    print(encoder)
    print("X_train: ", X_train.shape, " X_validation: ", X_validation.shape, " targets: ", targets.shape)

    # labels_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    # for idx, i in enumerate(y_train):
    #     labels_dict[i].append(idx)

    test_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for idx, i in enumerate(targets):
        test_dict[i].append(idx)

    for iteration in range(MAX_STEPS_DEFAULT):
        encoder.train()

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        p1, p2, mim = forward_block(X_train, ids, encoder, optimizer, train)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            encoder.eval()
            print("==================================================================================")
            print("ITERATION: ", iteration,
                  ",  batch size: ", BATCH_SIZE_DEFAULT,
                  ",  lr: ", LEARNING_RATE_DEFAULT,
                  ",  best loss iter: ", max_loss_iter,
                  "-", min_miss_percentage,
                  ",  actual best iter: ", most_clusters_iter,
                  "-", most_clusters,
                  ",  virtual best iter: ", best_virtual_iter,
                  ",", DESCRIPTION)

            miss_percentage, clusters, virtual_percentage = measure_acc_augments(X_validation, encoder, targets)

            if virtual_percentage < min_virtual_miss_percentage:
                best_virtual_iter = iteration
                min_virtual_miss_percentage = virtual_percentage
                print("models saved virtual iter: " + str(iteration))
                torch.save(encoder, virtual_best_path)

            if clusters >= most_clusters:
                min_miss_percentage = 1 - cluster_accuracies[clusters]
                most_clusters = clusters

                if min_miss_percentage > miss_percentage:
                    cluster_accuracies[clusters] = 1 - miss_percentage
                    max_loss_iter = iteration
                    most_clusters_iter = iteration

                    print("models saved iter: " + str(iteration))
                    torch.save(encoder, actual_best_path)

                    if clusters >= MIN_CLUSTERS_TO_SAVE:
                        for key in numbers_classes_dict.keys():
                            numpy_cluster = torch.zeros([len(numbers_classes_dict[key]), 1, SIZE, SIZE])
                            counter = 0
                            for index in numbers_classes_dict[key]:
                                numpy_cluster[counter] = orig_image.cpu().detach()[index]
                                counter += 1
                            if counter > 0:
                                save_cluster(numpy_cluster, key, iteration)


def to_tensor(X):
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


def print_info(p1, p2, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}
    #image_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    image_dict = {x: [] for x in range(CLASSES[0])}
    numbers_classes_dict = {x: [] for x in range(CLASSES[0])}

    cluster = BATCH_SIZE_DEFAULT // 10
    counter_cluster = 0

    for i in range(len(test_ids)):
        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)

        verdict = most_frequent([int(index.data.cpu().numpy()), int(index2.data.cpu().numpy())])

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + ", "

        # if i % cluster == 0:
        #     print("a ", labels_to_imags[counter_cluster], " 1: ", p1[i].data.cpu().numpy(), " ", "rc")
        #     print("a ", labels_to_imags[counter_cluster], " 2: ", p2[i].data.cpu().numpy(), " ", "rc hf")
        #     print("a ", labels_to_imags[counter_cluster], " 3: ", p3[i].data.cpu().numpy(), " ", "augment")
        #     print()
        #     counter_cluster += 1

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