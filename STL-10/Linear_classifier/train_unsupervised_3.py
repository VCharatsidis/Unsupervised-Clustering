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


EPS=sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000

BATCH_SIZE_DEFAULT = 100
INPUT_NET = 12544
SIZE = 44
NETS = 1
UNSUPERVISED = True
#DROPOUT = [0.90, 0.90, 0.90, 0.90, 0.90]
DROPOUT = [0.0, 0.0, 0.0, 0.0, 0.0]
class_n = 12
CLASSES = [class_n, class_n, class_n, class_n, class_n]
DESCRIPTION = " Image size: "+str(SIZE) + " , Dropout2d: "+str(DROPOUT)+" , Classes: "+str(CLASSES)

EVAL_FREQ_DEFAULT = 500
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
FLAGS = None


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


def encode_4_patches(image, encoder):
    # pad = (96 - SIZE) // 2
    # image /= 255
    #
    # horiz_f = horizontal_flip(image, BATCH_SIZE_DEFAULT)
    # original_image = scale(image, SIZE, pad, BATCH_SIZE_DEFAULT)
    # original_image = original_image[:, :, pad:96-pad, pad:96-pad]
    #
    # ids = random.randint(0, 9)
    #
    # image_3 = original_image
    #
    # if ids == 0:
    #     image_3 = scale(original_image, SIZE - 8, 4, BATCH_SIZE_DEFAULT)
    # elif ids == 1:
    #     rot = rotate(image, 24, BATCH_SIZE_DEFAULT)
    #     scale_rot = scale(rot, SIZE, pad, BATCH_SIZE_DEFAULT)
    #     image_3 = scale_rot[:, :, pad:96 - pad, pad:96 - pad]
    # elif ids == 2:
    #     rev_rot = rotate(image, -24, BATCH_SIZE_DEFAULT)
    #     scale_rev_rot = scale(rev_rot, SIZE, pad, BATCH_SIZE_DEFAULT)
    #     image_3 = scale_rev_rot[:, :, pad:96 - pad, pad:96 - pad]
    # elif ids == 3:
    #     image_3 = sobel_filter_x(original_image, BATCH_SIZE_DEFAULT)
    # elif ids == 4:
    #     image_3 = sobel_filter_y(original_image, BATCH_SIZE_DEFAULT)
    # elif ids == 5:
    #     image_3 = sobel_total(original_image, BATCH_SIZE_DEFAULT)
    # elif ids == 6:
    #     image_3 = torch.abs(1 - original_image)
    # elif ids == 7:
    #     image_soft_bin = binary(image)
    #     soft_bin = scale(image_soft_bin, SIZE, pad, BATCH_SIZE_DEFAULT)
    #     soft_bin = soft_bin[:, :, pad:96 - pad, pad:96 - pad]
    #     image_3 = torch.abs(1 - soft_bin)
    # elif ids == 8:
    #     image_3 = scale(original_image, SIZE - 12, 6, BATCH_SIZE_DEFAULT)
    #
    # image_1 = random_crop(image, SIZE, BATCH_SIZE_DEFAULT)
    # image_2 = random_crop(horiz_f, SIZE, BATCH_SIZE_DEFAULT)

    pad = (96 - SIZE) // 2
    image /= 255

    crop_preparation = scale(image, 72, 12, BATCH_SIZE_DEFAULT)
    crop_preparation = crop_preparation[:, :, 12:96 - 12, 12:96 - 12]
    crop_prep_horizontal = horizontal_flip(crop_preparation)

    image_soft_bin = binary(image)
    soft_bin = scale(image_soft_bin, SIZE, pad, BATCH_SIZE_DEFAULT)

    soft_bin = soft_bin[:, :, pad:96 - pad, pad:96 - pad]
    rev_soft_bin = torch.abs(1 - soft_bin)

    horiz_f = horizontal_flip(image, BATCH_SIZE_DEFAULT)
    soft_bin_hf = binary(horiz_f)
    soft_bin_hf = scale(soft_bin_hf, SIZE, pad, BATCH_SIZE_DEFAULT)
    soft_bin_hf = soft_bin_hf[:, :, pad:96 - pad, pad:96 - pad]

    rev_soft_bin_hf = torch.abs(1 - soft_bin_hf)

    rot = rotate(image, 24, BATCH_SIZE_DEFAULT)
    scale_rot = scale(rot, SIZE, pad, BATCH_SIZE_DEFAULT)
    scale_rot = scale_rot[:, :, pad:96 - pad, pad:96 - pad]

    rev_rot = rotate(image, -24, BATCH_SIZE_DEFAULT)
    scale_rev_rot = scale(rev_rot, SIZE, pad, BATCH_SIZE_DEFAULT)
    scale_rev_rot = scale_rev_rot[:, :, pad:96 - pad, pad:96 - pad]

    original_image = scale(image, SIZE, pad, BATCH_SIZE_DEFAULT)
    original_image = original_image[:, :, pad:96-pad, pad:96-pad]

    original_hfliped = horizontal_flip(original_image, BATCH_SIZE_DEFAULT)

    augments = {0: original_hfliped,
                1: scale(original_image, SIZE-8, 4, BATCH_SIZE_DEFAULT),
                2: scale_rot,
                3: scale_rev_rot,
                4: random_erease(original_image, BATCH_SIZE_DEFAULT),
                5: sobel_filter_x(original_image, BATCH_SIZE_DEFAULT),
                6: sobel_filter_y(original_image, BATCH_SIZE_DEFAULT),
                7: sobel_total(original_image, BATCH_SIZE_DEFAULT),
                8: sobel_filter_x(original_hfliped, BATCH_SIZE_DEFAULT),
                9: sobel_filter_y(original_hfliped, BATCH_SIZE_DEFAULT),
                10: sobel_total(original_hfliped, BATCH_SIZE_DEFAULT),
                # 11: binary(original_image),
                # 12: binary(horizontal_flip(original_image, BATCH_SIZE_DEFAULT)),
                # 13: binary(scale_rot),
                # 14: binary(scale_rev_rot),
                11: soft_bin,
                12: rev_soft_bin,
                13: soft_bin_hf,
                14: rev_soft_bin_hf,
                15: torch.abs(1 - original_image),
                16: rotate(original_hfliped, -15, BATCH_SIZE_DEFAULT),
                17: rotate(original_hfliped, 15, BATCH_SIZE_DEFAULT),
                18: scale(original_hfliped, SIZE - 10, 5, BATCH_SIZE_DEFAULT),
                19: scale(original_image, SIZE - 12, 6, BATCH_SIZE_DEFAULT),
                20: random_crop(crop_preparation, SIZE, BATCH_SIZE_DEFAULT),
                21: random_crop(crop_preparation, SIZE, BATCH_SIZE_DEFAULT),
                22: random_crop(crop_preparation, SIZE, BATCH_SIZE_DEFAULT),
                23: random_crop(crop_prep_horizontal, SIZE, BATCH_SIZE_DEFAULT),
                24: random_crop(crop_prep_horizontal, SIZE, BATCH_SIZE_DEFAULT),
                25: random_crop(crop_prep_horizontal, SIZE, BATCH_SIZE_DEFAULT),
                #26: original_image
                }

    ids = np.random.choice(len(augments), size=6, replace=False)

    image_1 = original_image # augments[ids[0]]
    image_2 = augments[ids[1]]
    image_3 = augments[ids[2]]

    # show_gray(image_1)
    # show_gray(image_2)
    # show_gray(image_3)

    _, test_preds_1, help_preds_1_1, help_preds_1_2,  help_preds_1_3, help_preds_1_4 = encoder(image_1.to('cuda'))
    _, test_preds_2, help_preds_2_1, help_preds_2_2,  help_preds_2_3, help_preds_2_4 = encoder(image_2.to('cuda'))
    _, test_preds_3, help_preds_3_1, help_preds_3_2,  help_preds_3_3, help_preds_3_4 = encoder(image_3.to('cuda'))

    return test_preds_1, test_preds_2, test_preds_3, \
           help_preds_1_1, help_preds_2_1,  help_preds_3_1, \
           help_preds_1_2, help_preds_2_2, help_preds_3_2, \
           help_preds_1_3, help_preds_2_3, help_preds_3_3, \
           help_preds_1_4, help_preds_2_4, help_preds_3_4, \
           original_image, ids


def entropy_minmax_loss(targets, preds_1, preds_2, preds_3):
    batch_cross_entropy_1 = batch_entropy(preds_1, targets)
    batch_cross_entropy_2 = batch_entropy(preds_2, targets)
    batch_cross_entropy_3 = batch_entropy(preds_3, targets)

    total_batch_cross_entropy = batch_cross_entropy_1 + batch_cross_entropy_2 + batch_cross_entropy_3

    product = preds_1 * preds_2 * preds_3
    product = product.mean(dim=0)
    log_product = torch.log(product)
    class_mean = -log_product.mean(dim=0)

    total_loss = class_mean - total_batch_cross_entropy

    return total_loss


def forward_block(X, ids, encoder, optimizer, train, total_mean):
    x_train = X[ids, :]
    x_train = rgb2gray(x_train)

    x_tensor = to_tensor(x_train)
    x_tensor = x_tensor.unsqueeze(0)
    images = x_tensor.transpose(0, 1)
    images = images.transpose(2, 3)

    test_preds_1, test_preds_2, test_preds_3, \
    help_preds_1_1, help_preds_2_1, help_preds_3_1, \
    help_preds_1_2, help_preds_2_2, help_preds_3_2, \
    help_preds_1_3, help_preds_2_3, help_preds_3_3, \
    help_preds_1_4, help_preds_2_4, help_preds_3_4,\
    orig_image, aug_ids = encode_4_patches(images, encoder)

    test_total_loss = entropy_minmax_loss(get_targets(CLASSES[0]), test_preds_1, test_preds_2, test_preds_3)
    help_total_loss_1 = entropy_minmax_loss(get_targets(CLASSES[1]), help_preds_1_1, help_preds_2_1, help_preds_3_1)
    help_total_loss_2 = entropy_minmax_loss(get_targets(CLASSES[2]), help_preds_1_2, help_preds_2_2, help_preds_3_2)
    help_total_loss_3 = entropy_minmax_loss(get_targets(CLASSES[3]), help_preds_1_3, help_preds_2_3, help_preds_3_3)
    help_total_loss_4 = entropy_minmax_loss(get_targets(CLASSES[4]), help_preds_1_4, help_preds_2_4, help_preds_3_4)

    m_preds = (test_preds_1 + test_preds_2 + test_preds_3) / 3
    total_mean = 0.99 * total_mean + 0.01 * m_preds.mean(dim=0).detach()

    all_losses = test_total_loss + help_total_loss_1 + help_total_loss_2 + help_total_loss_3 + help_total_loss_4

    if train:
        optimizer.zero_grad()
        all_losses.backward(retain_graph=True)
        optimizer.step()

    return test_preds_1, test_preds_2, test_preds_3, test_total_loss, total_mean, orig_image, aug_ids


def batch_entropy(pred, targets):
    batch_mean_preds = pred.mean(dim=0)
    H_batch = (targets.detach() * torch.log(batch_mean_preds)).sum()

    return H_batch


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images/iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def measure_acc_augments(X_test, colons, targets, total_mean):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    runs = len(X_test)//BATCH_SIZE_DEFAULT
    avg_loss = 0

    print()
    print("total mean:     ", total_mean.data.cpu().numpy())
    print()

    # augments = {0: "scale 8",
    #                    1: "rotate 24",
    #                    2: "rotate -24",
    #                    3: "sobel x",
    #                    4: "sobel y",
    #                    5: "sobel total",
    #                    6: "torch.abs(1 - original_image)",
    #                    7: "torch.abs(1 - soft_bin)",
    #                    8: "scale 12",
    #                     9: "scale 8",
    #                     10:"original"
    #                     }

    augments = {0: "horizontal flip",
                1: "scale",
                2: "rotate",
                3: "counter rotate",
                4: "random_erease",
                5: "sobel x",
                6: "sobel y",
                7: "sobel total",
                8: "sobel x fliped",
                9: "sobel y fliped",
                10: "sobel total fliped",
                # 11: "binary(original_image)",
                # 12: "binary(horizontal_flip)",
                # 13: "binary(scale_rot)",
                # 14: "binary(scale_rev_rot)",
                11: "soft_bin",
                12: "rev_soft_bin",
                13: "soft_bin_hf",
                14: "rev_soft_bin_hf",
                15: "torch.abs(1-original_image)",
                16: "rotate counter horizontal_flip",
                17: "rotate horizontal_flip",
                18: "scale hf 10",
                19: "scale 12",
                20: "random crop",
                21: "random crop",
                22: "random crop",
                23: "hf random crop",
                24: "hf random crop",
                25: "hf random crop",
                #26: "original"
                }

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        optimizers = []
        p1, p2, p3, mim, total_mean, orig_image, aug_ids = forward_block(X_test, test_ids, colons, optimizers, False, total_mean)

        if j == 0:
            print("a prediction 1: ", p1[0].data.cpu().numpy(), " ", augments[aug_ids[0]])
            print("a prediction 2: ", p2[0].data.cpu().numpy(), " ", augments[aug_ids[1]])
            print("a prediction 3: ", p3[0].data.cpu().numpy(), " ", augments[aug_ids[2]])

            print()

            print("a prediction 1: ", p1[20].data.cpu().numpy(), " ", augments[aug_ids[0]])
            print("a prediction 2: ", p2[20].data.cpu().numpy(), " ", augments[aug_ids[1]])
            print("a prediction 3: ", p3[20].data.cpu().numpy(), " ", augments[aug_ids[2]])

        avg_loss += mim.item()
        for i in range(p1.shape[0]):
            val, index = torch.max(p1[i], 0)
            val, index2 = torch.max(p2[i], 0)
            val, index3 = torch.max(p3[i], 0)

            preds = [index.data.cpu().numpy(), index2.data.cpu().numpy(), index3.data.cpu().numpy()]

            preds = list(preds)
            preds = [int(x) for x in preds]

            verdict = most_frequent(preds)

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
          " data: ", runs * BATCH_SIZE_DEFAULT,
          " miss percent: ", total_miss / (runs * BATCH_SIZE_DEFAULT))
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


def train():
    train_path = "..\\data\\stl10_binary\\train_X.bin"

    if UNSUPERVISED:
        train_path = "..\\data_2\\stl10_binary\\unlabeled_X.bin"

    print(train_path)
    X_train = read_all_images(train_path)

    # train_y_File = "..\\data\\stl10_binary\\train_y.bin"
    # y_train = read_labels(train_y_File)

    ########### test ##############################
    testFile = "..\\data\\stl10_binary\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "..\\data\\stl10_binary\\test_y.bin"
    targets = read_labels(test_y_File)

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'most_clusters_encoder' + '.model'
    clusters_net_path = os.path.join(script_directory, filepath)

    filepath = 'best_loss_encoder' + '.model'
    loss_net_path = os.path.join(script_directory, filepath)

    encoder = UnsupervisedNet(1, INPUT_NET, DROPOUT, CLASSES).to('cuda')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    max_loss = 1999
    max_loss_iter = 0
    most_clusters = 0
    most_clusters_iter = 0

    print(encoder)
    print("X_train: ", X_train.shape, " X_test: ", X_test.shape, " targets: ", targets.shape)
    total_mean = torch.ones([CLASSES[0]]) * (1/CLASSES[0])
    total_mean = total_mean.to('cuda')

    # labels_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    # for idx, i in enumerate(y_train):
    #     labels_dict[i].append(idx)

    test_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    for idx, i in enumerate(targets):
        test_dict[i].append(idx)

    for iteration in range(MAX_STEPS_DEFAULT):
        encoder.train()
        #ids = []
        samples_per_cluster = BATCH_SIZE_DEFAULT // 10
        #
        # for i in range(1, 11):
        #     ids += random.sample(labels_dict[i], samples_per_cluster)

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        p1, p2, p3, mim, total_mean, orig_image, aug_ids = forward_block(X_train, ids, encoder, optimizer, train, total_mean)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            encoder.eval()
            print("==================================================================================")
            print("ITERATION: ", iteration,
                  ",  batch size: ", BATCH_SIZE_DEFAULT,
                  ",  lr: ", LEARNING_RATE_DEFAULT,
                  ",  Unsupervised: ", UNSUPERVISED,
                  ",  best loss iter: ", max_loss_iter,
                  "-", max_loss,
                  ",  most clusters iter: ", most_clusters_iter,
                  "-", most_clusters,
                  ",", DESCRIPTION)

            test_ids = []

            for i in range(1, 11):
                test_ids += random.sample(test_dict[i], samples_per_cluster)

            p1, p2, p3, mim, total_mean, orig_image, aug_ids = forward_block(X_test, test_ids, encoder, optimizer, False, total_mean)
            classes_dict, numbers_classes_dict = print_info(p1, p2, p3, targets, test_ids)

            loss, clusters = measure_acc_augments(X_test, encoder, targets, total_mean)

            if clusters >= MIN_CLUSTERS_TO_SAVE:
                for key in numbers_classes_dict.keys():
                    numpy_cluster = torch.zeros([len(numbers_classes_dict[key]), 1, SIZE, SIZE])
                    counter = 0
                    for index in numbers_classes_dict[key]:
                        numpy_cluster[counter] = orig_image.cpu().detach()[index]
                        counter += 1
                    if counter > 0:
                        save_cluster(numpy_cluster, key, iteration)

                # for i in image_dict.keys():
                #     for index in image_dict[i]:
                #         save_image(orig_image.cpu().detach()[index], index, "iter_"+str(iteration)+"_"+labels_to_imags[targets[test_ids[index]]], i)

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


def print_info(p1, p2, p3, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}
    #image_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    image_dict = {x: [] for x in range(CLASSES[0])}
    numbers_classes_dict = {x: [] for x in range(CLASSES[0])}

    for i in range(len(test_ids)):
        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)
        val, index3 = torch.max(p3[i], 0)

        verdict = most_frequent([int(index.data.cpu().numpy()), int(index2.data.cpu().numpy()), int(index3.data.cpu().numpy())])

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + " " + str(
            index3.data.cpu().numpy()) + ", "

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