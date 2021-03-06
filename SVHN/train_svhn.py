from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import scipy.io as sio
import argparse
import os

from svhn_encoder import SVHNencoderNet
from another_encoder import SVHNComiteeEncoder
from stl_utils import *
import random

from torchvision.utils import make_grid
import matplotlib
from torchvision import models
EPS = sys.float_info.epsilon


fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

#EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 220

INPUT_NET = 4608
SIZE = 32
SIZE_Y = 20

DROPOUT = 0
class_n = 10
CLASSES = [class_n, class_n, class_n, class_n, class_n]
DESCRIPTION = " Image size: "+str(SIZE) + " , Dropout2d: "+str(DROPOUT)+" , Classes: "+str(CLASSES)

EVAL_FREQ_DEFAULT = 500
MIN_CLUSTERS_TO_SAVE = 9
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
FLAGS = None


cluster_accuracies = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

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


def transformation(id, image):
    if id == 0:
        return color_jitter(image)
    elif id == 1:
        return scale(color_jitter(image), (image.shape[2] - 8, image.shape[3] - 8), 4, BATCH_SIZE_DEFAULT)
    elif id == 2:
        return rotate(color_jitter(image), 30)
    elif id == 3:
        return torch.abs(1 - color_jitter(image))
    elif id == 4:
        return sobel_total(color_jitter(image), BATCH_SIZE_DEFAULT)
    elif id == 5:
        return sobel_filter_x(color_jitter(image), BATCH_SIZE_DEFAULT)

    elif id == 6:
        return sobel_filter_y(color_jitter(image), BATCH_SIZE_DEFAULT)

    # elif id == 7:
    #     return horizontal_flip(color_jitter(image), BATCH_SIZE_DEFAULT)

    elif id == 7:
        return gaussian_blur(color_jitter(image))

    elif id == 8:
        t = random_crop(color_jitter(image), 22, BATCH_SIZE_DEFAULT, 18)
        return scale_up(t, 32, 20, BATCH_SIZE_DEFAULT)

    print("Error in transformation of the image.")
    return image


def entropy_minmax_loss(preds_1, preds_2, preds_3, preds_4, total_mean):
    product = preds_1 * preds_2 * preds_3 * preds_4
    product = product.mean(dim=0) #* total_mean.detach()
    #product[(product < EPS).data] = EPS
    log_product = torch.log(product)
    total_loss = - log_product.mean(dim=0)

    return total_loss


def forward_block(X_train, ids, encoder, optimizer, train, total_mean):
    if train:
        optimizer.zero_grad()

    scaled_loss = 0
    for accumulated_step_i in range(2):
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        image = X_train[ids, :]

        number_transforms = 9
        aug_ids = np.random.choice(number_transforms, size=number_transforms, replace=False)

        image_1 = transformation(aug_ids[0], image)
        image_2 = transformation(aug_ids[1], image)
        image_3 = transformation(aug_ids[2], image)
        image_4 = transformation(aug_ids[3], image)

        _, _, test_preds_1 = encoder(image_1.to('cuda'))
        _, _, test_preds_2 = encoder(image_2.to('cuda'))
        _, _, test_preds_3 = encoder(image_3.to('cuda'))
        _, _, test_preds_4 = encoder(image_4.to('cuda'))

        # show_gray(image_1)
        # show_gray(image_2)
        # show_gray(image_3)
        # show_gray(image_4)


        test_total_loss = entropy_minmax_loss(test_preds_1, test_preds_2, test_preds_3, test_preds_4, total_mean)

        m_preds = (test_preds_1 + test_preds_2 + test_preds_3 + test_preds_4) / 4
        total_mean = 0.99 * total_mean + 0.01 * m_preds.mean(dim=0).detach()

        test_total_loss.backward()
        scaled_loss += test_total_loss.item()

    if train:
        optimizer.step()

    return test_preds_1, test_preds_2, test_preds_3, test_preds_4, test_total_loss, total_mean, image_1, ids


def batch_entropy(pred, targets):
    batch_mean_preds = pred.mean(dim=0)
    H_batch = torch.log(batch_mean_preds).mean()

    return H_batch


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"new_4_images/iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"new_4_images/c_{cluster}/{name}_index_{index}.png", sample)


def accuracy_block(X, ids, encoder):
    images = X[ids, :]
    #sobel = transformation(4, images)
    _, _, test_preds_1 = encoder(images.to('cuda'))

    return test_preds_1

def measure_acc_augments(X_test, colons, targets, total_mean):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    size = BATCH_SIZE_DEFAULT
    runs = len(X_test)//size
    avg_loss = 0

    print()
    print("total mean:     ", total_mean.data.cpu().numpy())
    print()

    batch_num = random.randint(0, runs - 1)

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)
        p = accuracy_block(X_test, test_ids, colons)

        if j == batch_num:
            pred_number = random.randint(0, BATCH_SIZE_DEFAULT - 1)
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


def preproccess(data):
    data = to_tensor(data)

    data = data.transpose(0, 2)
    data = data.transpose(1, 3)
    data = data.transpose(0, 1)

    pad = (SIZE-SIZE_Y) // 2
    data = data[:, :, :, pad: SIZE - pad]
    data = rgb2gray(data)

    data = data.unsqueeze(0)
    data = data.transpose(0, 1)
    data /= 255

    return data


def train():
    unsupervised_data = sio.loadmat('data\\extra_32x32.mat')
    train_data = sio.loadmat('data\\train_32x32.mat')
    test_data = sio.loadmat('data\\test_32x32.mat')

    X_train = preproccess(unsupervised_data['X'])
    print("x train shape", X_train.shape)

    # access to the dict

    X_test = preproccess(test_data['X'])
    print("x test shape", X_test.shape)

    targets = test_data['y'].squeeze(1)
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

    for iteration in range(MAX_STEPS_DEFAULT):
        encoder.train()
        samples_per_cluster = BATCH_SIZE_DEFAULT // 10

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        p1, p2, p3, p4, mim, total_mean, orig_image, aug_ids = forward_block(X_train, ids, encoder, optimizer, train, total_mean)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            encoder.eval()
            print("==================================================================================")
            print("ITERATION: ", iteration,
                  ",  batch size: ", BATCH_SIZE_DEFAULT,
                  ",  lr: ", LEARNING_RATE_DEFAULT,
                  ",  best loss iter: ", max_loss_iter,
                  "-", min_miss_percentage,
                  ",  most clusters iter: ", most_clusters_iter,
                  "-", most_clusters,
                  ",", DESCRIPTION)

            test_ids = []

            for i in range(0, 10):
                test_ids += random.sample(test_dict[i], samples_per_cluster)

            p1, p2, p3, p4, mim, total_mean, orig_image, test_ids = forward_block(X_test, test_ids, encoder, optimizer, False, total_mean)
            classes_dict, numbers_classes_dict = print_info(p1, p2, p3, p4, targets, test_ids)

            miss_percentage, clusters = measure_acc_augments(X_test, encoder, targets, total_mean)

            if clusters >= most_clusters:
                min_miss_percentage = 1 - cluster_accuracies[clusters]
                most_clusters = clusters

                if min_miss_percentage > miss_percentage:
                    cluster_accuracies[clusters] = 1 - miss_percentage
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


def print_info(p1, p2, p3, p4, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}
    #image_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    image_dict = {x: [] for x in range(CLASSES[0])}
    numbers_classes_dict = {x: [] for x in range(CLASSES[0])}

    for i in range(len(test_ids)):
        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)
        val, index3 = torch.max(p3[i], 0)
        val, index4 = torch.max(p4[i], 0)

        verdict = most_frequent([int(index.data.cpu().numpy()), int(index2.data.cpu().numpy()), int(index3.data.cpu().numpy()), int(index4.data.cpu().numpy())])

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + " " + str(
            index3.data.cpu().numpy()) + " " + str(index4.data.cpu().numpy()) +", "

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