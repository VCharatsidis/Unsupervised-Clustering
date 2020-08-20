from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import scipy.io as sio
import argparse
import os

from multy_head_dropout_encoder import MultiHeadDropout

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

BATCH_SIZE_DEFAULT = 580

#INPUT_NET = 3072
INPUT_NET = 5120
SIZE = 32
SIZE_Y = 20
NETS = 1

DROPOUT = 0
class_n = 10
CLASSES = [class_n, class_n, class_n, class_n, class_n]
DESCRIPTION = " Image size: "+str(SIZE) + " , Dropout2d: "+str(DROPOUT)+" , Classes: "+str(CLASSES)

EVAL_FREQ_DEFAULT = 500
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
    half = BATCH_SIZE_DEFAULT//2
    if id == 0:
        return color_jitter(image)
    elif id == 1:
        return scale(color_jitter(image), (image.shape[2] - 8, image.shape[3] - 8), 4, half)
    elif id == 2:
        return rotate(color_jitter(image), 30)
    elif id == 3:
        return torch.abs(1 - color_jitter(image))
    elif id == 4:
        return sobel_total(color_jitter(image), half)
    elif id == 5:
        return sobel_filter_x(color_jitter(image), half)
    elif id == 6:
        return sobel_filter_y(color_jitter(image), half)
    elif id == 7:
        return gaussian_blur(color_jitter(image))

    elif id == 8:
        t = random_crop(color_jitter(image), 22, half, 18)
        scaled_up = scale_up(t, 32, 20, half)
        blured = gaussian_blur(scaled_up)
        return blured

    elif id == 9:
        t = random_crop(color_jitter(image), 22, half, 18)
        scaled_up = scale_up(t, 32, 20, half)
        sobeled = sobel_total(scaled_up, half)
        return sobeled

    elif id == 10:
        t = random_crop(color_jitter(image), 22, half, 18)
        scaled_up = scale_up(t, 32, 20, half)
        rev = torch.abs(1 - scaled_up)
        return rev

    elif id == 11:
        t = random_crop(color_jitter(image), 22, half, 18)
        scaled_up = scale_up(t, 32, 20, half)
        rot = rotate(scaled_up, 40)
        return rot

    elif id == 11:
        return horizontal_flip(color_jitter(image), half)



    print("Error in transformation of the image.")
    return image


def entropy_minmax_loss(preds_1, preds_2, total_mean):
    product = preds_1 * preds_2
    product = product.mean(dim=0)  # * total_mean.detach()
    #product[(product < EPS).data] = EPS
    class_logs = torch.log(product) #* cuda_tensor_priors
    total_loss = - class_logs.mean(dim=0)

    return total_loss


def forward_block(X, ids, encoder, optimizer, train, total_mean):
    number_transforms = 12
    aug_ids = np.random.choice(number_transforms, size=number_transforms, replace=False)

    ids = np.random.choice(len(X), size=BATCH_SIZE_DEFAULT, replace=False)
    image = X[ids, :]

    half = BATCH_SIZE_DEFAULT // 2
    image_1 = transformation(aug_ids[0], image[0:half])
    image_2 = transformation(aug_ids[1], image[0:half])

    image_3 = transformation(aug_ids[2], image[half:])
    image_4 = transformation(aug_ids[3], image[half:])

    image_1 = torch.cat([image_1, image_3], dim=0)
    image_2 = torch.cat([image_2, image_4], dim=0)

    encoding, probs10, probs12, probs20, probs50 = encoder(image_1.to('cuda'))
    encoding, probs10_b, probs12_b, probs20_b, probs50_b = encoder(image_2.to('cuda'))

    loss10 = entropy_minmax_loss(probs10, probs10_b, total_mean)
    loss12 = entropy_minmax_loss(probs12, probs12_b, total_mean)
    loss20 = entropy_minmax_loss(probs20, probs20_b, total_mean)
    loss50 = entropy_minmax_loss(probs50, probs50_b, total_mean)

    total_loss = loss10 + loss12 + loss20 + loss50

    if train:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return probs10, probs10_b, total_loss, total_mean, image, ids


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
    encoding, probs10, probs12, probs20, probs50 = encoder(images.to('cuda'))

    return probs10


def measure_acc_augments(X_test, colons, targets, total_mean):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    size = BATCH_SIZE_DEFAULT
    runs = len(X_test)//size
    avg_loss = 0

    print()
    print("total mean:     ", total_mean.data.cpu().numpy())
    print()
    batch_num = random.randint(0, runs-1)

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)

        p = accuracy_block(X_test, test_ids, colons)

        if j == batch_num:
            pred_number = random.randint(0, BATCH_SIZE_DEFAULT-1)
            print("example prediction ", pred_number, " : ", p[pred_number].data.cpu().numpy())
            pred_number = random.randint(0, BATCH_SIZE_DEFAULT - 1)
            print("example prediction ", pred_number, " : ", p[pred_number].data.cpu().numpy())
            pred_number = random.randint(0, BATCH_SIZE_DEFAULT - 1)
            print("example prediction ", pred_number, " : ", p[pred_number].data.cpu().numpy())
            pred_number = random.randint(0, BATCH_SIZE_DEFAULT - 1)
            print("example prediction ", pred_number, " : ", p[pred_number].data.cpu().numpy())
            pred_number = random.randint(0, BATCH_SIZE_DEFAULT - 1)
            print("example prediction ", pred_number, " : ", p[pred_number].data.cpu().numpy())

        for i in range(p.shape[0]):
            _, mean_index = torch.max(p[i], 0)
            verdict = int(mean_index.data.cpu().numpy())
            label = targets[test_ids[i]]

            if label == 10:
                label = 0

            print_dict[label].append(verdict)

        # p1, p2, mim, total_mean, orig_image, aug_ids = forward_block(X_test, test_ids, colons, optimizers, False, total_mean)
        # avg_loss += mim.item()
        # for i in range(p1.shape[0]):
        #     mean = (p1[i] + p2[i]) / 2
        #     _, mean_index = torch.max(mean, 0)
        #     verdict = int(mean_index.data.cpu().numpy())
        #
        #     label = targets[test_ids[i]]
        #     if label == 10:
        #         label = 0
        #
        #     print_dict[label].append(verdict)

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
    print("AUGMENTS avg loss: ", avg_loss / runs,
          " miss: ", total_miss,
          " data: ", runs * size,
          " miss percent: ", miss_percentage)
    print("Clusters found: " + str(len(clusters)) + " " + str(clusters))
    print()

    return miss_percentage, len(clusters)


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
    unsupervised_data = sio.loadmat('data\\extra_32x32.mat')
    train_data = sio.loadmat('data\\train_32x32.mat')
    test_data = sio.loadmat('data\\test_32x32.mat')

    original_train = preproccess(train_data['X'])
    train_targets = train_data['y'].squeeze(1)
    print("y test", train_targets.shape)

    train_targets = np.array([x % 10 for x in train_targets])

    # for target in train_targets:
    #     class_numbers[target] += 1

    X_train = preproccess(unsupervised_data['X'])
    print("x train shape", X_train.shape)

    # access to the dict

    X_test = preproccess(test_data['X'])
    print("x test shape", X_test.shape)

    targets = test_data['y'].squeeze(1)
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

    filepath = 'svhn_models\\most_clusters_encoder' + '.model'
    clusters_net_path = os.path.join(script_directory, filepath)

    filepath = 'svhn_models\\best_loss_encoder' + '.model'
    loss_net_path = os.path.join(script_directory, filepath)

    encoder = MultiHeadDropout(1, INPUT_NET, DROPOUT, CLASSES).to('cuda')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    min_miss_percentage = 100
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

    avg_loss = 0
    for iteration in range(MAX_STEPS_DEFAULT):
        encoder.train()
        samples_per_cluster = BATCH_SIZE_DEFAULT // 10

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        p1, p2, mim, total_mean, orig_image, aug_ids = forward_block(X_train, ids, encoder, optimizer, train, total_mean)
        avg_loss += mim.item()
        if iteration % EVAL_FREQ_DEFAULT == 0:
            print("==================================================================================")
            print("avg loss : ", avg_loss/(iteration+1))
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

            test_ids = np.random.choice(len(original_train), size=BATCH_SIZE_DEFAULT, replace=False)
            p1, p2, mim, total_mean, orig_image, test_ids = forward_block(original_train, test_ids, encoder, optimizer, False, total_mean)
            classes_dict, numbers_classes_dict = print_info(p1, p2, train_targets, test_ids)

            miss_percentage, clusters = measure_acc_augments(X_test, encoder, targets, total_mean)

            if clusters >= most_clusters:
                min_miss_percentage = 1 - best_cluster_accuracies[clusters]
                most_clusters = clusters

                if min_miss_percentage > miss_percentage:
                    best_cluster_accuracies[clusters] = 1 - miss_percentage
                    min_miss_percentage = miss_percentage
                    max_loss_iter = iteration
                    most_clusters_iter = iteration

                    print("models saved iter: " + str(iteration))
                    torch.save(encoder, clusters_net_path)

                    if clusters >= MIN_CLUSTERS_TO_SAVE:
                        for key in numbers_classes_dict.keys():
                            numpy_cluster = torch.zeros([len(numbers_classes_dict[key]), 1, SIZE, SIZE_Y])
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