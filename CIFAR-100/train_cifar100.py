from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import scipy.io as sio
import argparse
import os

from cifarNet import CifarNet

from stl_utils import *
import random

from torchvision.utils import make_grid
import matplotlib
import pickle

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)

from torchvision import models
EPS = sys.float_info.epsilon

#EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4

MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 300

#INPUT_NET = 3072
INPUT_NET = 5120
SIZE = 32
SIZE_Y = 32
NETS = 1

CLASSES = 100
DESCRIPTION = " Image size: " + str(SIZE) + " , Classes: " + str(CLASSES)

EVAL_FREQ_DEFAULT = 500
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None


cluster_accuracies = {}
for i in range(CLASSES):
    cluster_accuracies[i] = 0


class_numbers = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 0: 0}


transformations_dict = {0: "original",
                       1: "scale",
                       2: "rotate",
                       3: "reverse pixel value",
                       4: "sobel total",
                       5: "sobel x",
                       6: "sobel y",
                       7: "gaussian blur",
                       8: "random crop",
                       9: "random crop",
                       10: "random crop",
                       11: "random crop",
                       12: "image 1",
                       13: "image 2"}


labels_to_imags = {}
for i in range(CLASSES):
    labels_to_imags[i] = i


def save_images(images, transformation):
    print(transformations_dict[transformation])
    numpy_cluster = images.cpu().detach()
    save_cluster(numpy_cluster, transformations_dict[transformation], 0)


def transformation(id, image):
    quarter = BATCH_SIZE_DEFAULT//4

    if random.uniform(0, 1) > 0.5:
        image = horizontal_flip(image, quarter)

    if id == 0:
        return color_jitter(image)
    elif id == 1:
        return scale(color_jitter(image), (image.shape[2] - 8, image.shape[3] - 8), 4, quarter)
    elif id == 2:
        return rotate(color_jitter(image), 30)
    elif id == 3:
        return torch.abs(1 - color_jitter(image))
    elif id == 4:
        sobeled = sobel_total(color_jitter(image), quarter)

        AA = sobeled.view(sobeled.size(0), -1)
        AA -= AA.min(1, keepdim=True)[0]
        AA /= AA.max(1, keepdim=True)[0]
        AA = AA.view(quarter, 1, SIZE, SIZE)

        return AA

    elif id == 5:
        sobeled = sobel_filter_x(color_jitter(image), quarter)

        AA = sobeled.view(sobeled.size(0), -1)
        AA -= AA.min(1, keepdim=True)[0]
        AA /= AA.max(1, keepdim=True)[0]
        AA = AA.view(quarter, 1, SIZE, SIZE)

        return AA

    elif id == 6:
        sobeled = sobel_filter_y(color_jitter(image), quarter)

        AA = sobeled.view(sobeled.size(0), -1)
        AA -= AA.min(1, keepdim=True)[0]
        AA /= AA.max(1, keepdim=True)[0]
        AA = AA.view(quarter, 1, SIZE, SIZE)

        return AA

    elif id == 7:
        return gaussian_blur(color_jitter(image))

    elif id == 8:
        t = random_crop(color_jitter(image), 22, quarter, 22)
        scaled_up = scale_up(t, 32, 32, quarter)
        blured = gaussian_blur(scaled_up)
        return blured

    elif id == 9:
        t = random_crop(color_jitter(image), 22, quarter, 22)
        scaled_up = scale_up(t, 32, 32, quarter)
        sobeled = sobel_total(scaled_up, quarter)

        AA = sobeled.view(sobeled.size(0), -1)
        AA -= AA.min(1, keepdim=True)[0]
        AA /= AA.max(1, keepdim=True)[0]
        AA = AA.view(quarter, 1, SIZE, SIZE)

        return AA


    elif id == 10:
        t = random_crop(color_jitter(image), 22, quarter, 22)
        scaled_up = scale_up(t, 32, 32, quarter)
        rev = torch.abs(1 - scaled_up)
        return rev

    elif id == 11:
        t = random_crop(color_jitter(image), 22, quarter, 22)
        scaled_up = scale_up(t, 32, 32, quarter)
        rot = rotate(scaled_up, 40)
        return rot

    print("Error in transformation of the image.")
    return image


def entropy_minmax_loss(preds_1, preds_2, total_mean):
    product = preds_1 * preds_2
    product = product.mean(dim=0)

    coeff = 0.8
    # product = coeff * product + (1 - coeff) * total_mean.to('cuda')
    total_mean = coeff * total_mean + (1 - coeff) * product.detach().to('cpu')

    class_logs = torch.log(product)
    total_loss = - class_logs.mean(dim=0)

    return total_loss, total_mean


def forward_block(X, ids, encoder, optimizer, train, total_mean):
    number_transforms = 12
    aug_ids = np.random.choice(number_transforms, size=number_transforms, replace=False)

    ids = np.random.choice(len(X), size=BATCH_SIZE_DEFAULT, replace=False)
    image = X[ids, :]

    show_gray(image)

    fourth = BATCH_SIZE_DEFAULT // 4
    image_1 = transformation(aug_ids[0], image[0:fourth])
    image_2 = transformation(aug_ids[1], image[0:fourth])

    image_3 = transformation(aug_ids[2], image[fourth: 2 * fourth])
    image_4 = transformation(aug_ids[3], image[fourth: 2 * fourth])

    image_5 = transformation(aug_ids[4], image[2 * fourth: 3 * fourth])
    image_6 = transformation(aug_ids[5], image[2 * fourth: 3 * fourth])

    image_7 = transformation(aug_ids[6], image[3 * fourth:])
    image_8 = transformation(aug_ids[7], image[3 * fourth:])

    save_images(image_1, aug_ids[0])
    save_images(image_2, aug_ids[1])
    save_images(image_3, aug_ids[2])
    save_images(image_4, aug_ids[3])
    save_images(image_5, aug_ids[4])
    save_images(image_6, aug_ids[5])
    save_images(image_7, aug_ids[6])
    save_images(image_8, aug_ids[7])

    image_1 = torch.cat([image_1, image_3, image_5, image_7], dim=0)
    image_2 = torch.cat([image_2, image_4, image_6, image_8], dim=0)

    save_images(image_1, 12)
    save_images(image_2, 13)

    encoding, probs10 = encoder(image_1.to('cuda'))
    encoding, probs10_b = encoder(image_2.to('cuda'))

    total_loss, total_mean = entropy_minmax_loss(probs10, probs10_b, total_mean)

    if train:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return probs10, probs10_b, total_loss, total_mean


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def measure_acc_augments(X_test, encoder, targets):
    virtual_clusters = {}
    for i in range(CLASSES):
        virtual_clusters[i] = []

    size = BATCH_SIZE_DEFAULT
    runs = len(X_test)//size
    avg_loss = 0

    print_dict = {}
    virtual_clusters = {}
    for i in range(CLASSES):
        print_dict[i] = []
        virtual_clusters[i] = []

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)

        images = X_test[test_ids, :]

        _, p = encoder(images.to('cuda'))

        for i in range(p.shape[0]):
            val, index = torch.max(p[i], 0)
            verdict = int(index.data.cpu().numpy())

            label = targets[test_ids[i]]
            # if label == 10:
            #     label = 0

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


def preproccess_cifar(x):
    x = to_tensor(x)

    x = x.transpose(1, 3)

    pad = (SIZE-SIZE_Y) // 2
    x = x[:, :, :, pad: SIZE - pad]
    x = rgb2gray(x)

    x = x.unsqueeze(0)
    x = x.transpose(0, 1)
    x /= 255

    return x

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def train():
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

    filepath = 'cifar100_models\\actual_best' + '.model'
    actual_best_path = os.path.join(script_directory, filepath)

    filepath = 'cifar100_models\\virtual_best' + '.model'
    virtual_best_path = os.path.join(script_directory, filepath)

    encoder = CifarNet(1, CLASSES).to('cuda')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    min_miss_percentage = 100
    max_loss_iter = 0
    most_clusters = 0
    most_clusters_iter = 0

    best_virtual_iter = 0
    min_virtual_miss_percentage = 1000

    print(encoder)
    print("X_train: ", X_train.shape, " X_test: ", X_test.shape, " targets: ", targets.shape)
    total_mean = torch.ones(CLASSES) * (1 / CLASSES)

    # test_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    # for idx, i in enumerate(targets):
    #     test_dict[i].append(idx)

    avg_loss = 0
    for iteration in range(MAX_STEPS_DEFAULT):
        encoder.train()

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        probs10, probs10_b, total_loss, total_mean = forward_block(X_train, ids, encoder, optimizer, train, total_mean)
        avg_loss += total_loss.item()

        if iteration % EVAL_FREQ_DEFAULT == 0:
            print("==================================================================================")
            print("train avg loss : ", avg_loss / BATCH_SIZE_DEFAULT)
            avg_loss = 0
            encoder.eval()
            print("total mean: ", total_mean)

            print("ITERATION: ", iteration,
                  ",  batch size: ", BATCH_SIZE_DEFAULT,
                  ",  lr: ", LEARNING_RATE_DEFAULT,
                  ",  best loss iter: ", max_loss_iter,
                  "-", min_miss_percentage,
                  ",  actual best iter: ", most_clusters_iter,
                  "-", most_clusters,
                  ",  virtual best iter: ", best_virtual_iter,
                  ",", DESCRIPTION)

            # test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            # probs10, probs10_b, total_loss, total_mean = forward_block(X_test, test_ids, encoder, optimizer, False, total_mean)

            miss_percentage, clusters, virtual_percentage = measure_acc_augments(X_test, encoder, targets)

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