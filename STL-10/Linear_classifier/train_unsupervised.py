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

BATCH_SIZE_DEFAULT = 40
INPUT_NET = 8192
SIZE = 44
NETS = 1
DESCRIPTION = "Augments: 3 augments then 3 sobels x,y,total. Nets: "+str(NETS) +" net. Loss: total_loss = paired_losses - mean_probs_losses. Image size: " + str(SIZE)

EVAL_FREQ_DEFAULT = 200
NUMBER_CLASSES = 10
MIN_CLUSTERS_TO_SAVE = 9
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
FLAGS = None

TARGETS = (torch.ones([NUMBER_CLASSES]).to('cuda')) / NUMBER_CLASSES


def encode_4_patches(image, encoder):
    pad = (96 - SIZE) // 2
    image /= 255

    rot = rotate(image, 20, BATCH_SIZE_DEFAULT)
    scale_rot = scale(rot, SIZE, pad, BATCH_SIZE_DEFAULT)
    scale_rot = scale_rot[:, :, pad:96 - pad, pad:96 - pad]

    rev_rot = rotate(image, -20, BATCH_SIZE_DEFAULT)
    scale_rev_rot = scale(rev_rot, SIZE, pad, BATCH_SIZE_DEFAULT)
    scale_rev_rot = scale_rev_rot[:, :, pad:96 - pad, pad:96 - pad]

    original_image = scale(image, SIZE, pad, BATCH_SIZE_DEFAULT)
    original_image = original_image[:, :, pad:96-pad, pad:96-pad]

    augments = {0: horizontal_flip(original_image, BATCH_SIZE_DEFAULT),
                1: scale(original_image, SIZE-8, 4, BATCH_SIZE_DEFAULT),
                2: scale_rot,
                3: scale_rev_rot,
                4: random_erease(original_image, BATCH_SIZE_DEFAULT),
                5: sobel_filter_x(original_image, BATCH_SIZE_DEFAULT),
                6: sobel_filter_y(original_image, BATCH_SIZE_DEFAULT),
                7: sobel_total(original_image, BATCH_SIZE_DEFAULT),
                8: sobel_filter_x(horizontal_flip(original_image, BATCH_SIZE_DEFAULT), BATCH_SIZE_DEFAULT),
                9: sobel_filter_y(horizontal_flip(original_image, BATCH_SIZE_DEFAULT), BATCH_SIZE_DEFAULT),
                10: sobel_total(horizontal_flip(original_image, BATCH_SIZE_DEFAULT), BATCH_SIZE_DEFAULT),
                11: binary(original_image),
                12: binary(horizontal_flip(original_image, BATCH_SIZE_DEFAULT)),
                13: binary(scale_rot),
                14: binary(scale_rev_rot),
                #16: center_crop(image, SIZE, BATCH_SIZE_DEFAULT),
                }

    ids = np.random.choice(len(augments), size=6, replace=False)

    image_1 = original_image
    image_2 = augments[ids[1]]
    image_3 = augments[ids[2]]
    image_4 = augments[ids[3]]
    image_5 = augments[ids[4]]
    image_6 = augments[ids[5]]

    # show_gray(image_1)
    # show_gray(image_2)
    # show_gray(image_3)
    # show_gray(image_4)
    # show_gray(image_5)
    # show_gray(image_6)

    _, preds_1 = encoder(image_1.to('cuda'))
    _, preds_2 = encoder(image_2.to('cuda'))
    _, preds_3 = encoder(image_3.to('cuda'))
    _, preds_4 = encoder(image_4.to('cuda'))
    _, preds_5 = encoder(image_5.to('cuda'))
    _, preds_6 = encoder(image_6.to('cuda'))

    return preds_1, preds_2, preds_3, preds_4, preds_5, preds_6, original_image, ids


def forward_block(X, ids, encoder, optimizer, train, total_mean):

    x_train = X[ids, :]
    x_train = rgb2gray(x_train)

    x_tensor = to_tensor(x_train)
    x_tensor = x_tensor.unsqueeze(0)
    images = x_tensor.transpose(0, 1)
    images = images.transpose(2, 3)

    preds_1, preds_2, preds_3, preds_4, preds_5, preds_6, orig_image, aug_ids = encode_4_patches(images, encoder)

    batch_pred_1 = batch_entropy(preds_1)
    batch_pred_2 = batch_entropy(preds_2)
    batch_pred_3 = batch_entropy(preds_3)
    batch_pred_4 = batch_entropy(preds_4)
    batch_pred_5 = batch_entropy(preds_5)
    batch_pred_6 = batch_entropy(preds_6)

    batch_loss = batch_pred_1 + batch_pred_2 + batch_pred_3 + batch_pred_4 + batch_pred_5 + batch_pred_6

    product = preds_1 * preds_2 * preds_3 * preds_4 * preds_5 * preds_6
    mean = product.mean(dim=0)
    log = torch.log(mean + EPS)
    total_loss = - (TARGETS * log).mean() - batch_loss

    m_preds = (preds_1 + preds_2 + preds_3 + preds_4 + preds_5 + preds_6) / 6
    total_mean = 0.99 * total_mean + 0.01 * m_preds.mean(dim=0).detach()

    if train:
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

    return preds_1, preds_2, preds_3, preds_4, preds_5, preds_6, total_loss, total_mean, orig_image, aug_ids


def batch_entropy(pred):
    batch_mean_preds = pred.mean(dim=0)
    H_batch = (TARGETS.detach() * torch.log(batch_mean_preds + EPS)).sum()

    return H_batch


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{iteration}_cluster_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/c_{cluster}/{name}_index_{index}.png", sample)


def measure_acc_augments(X_test, colons, targets, total_mean):
    print_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    runs = len(X_test)//BATCH_SIZE_DEFAULT
    avg_loss = 0

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
                11: "binary(original_image)",
                12: "binary(horizontal_flip)",
                13: "binary(scale_rot)",
                14: "binary(scale_rev_rot)",
                #16: "center_crop",
                }

    print("total mean:     ", total_mean.data.cpu().numpy())
    print()

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        optimizers = []
        p1, p2, p3, p4, p5, p6, mim, total_mean, orig_image, aug_ids = forward_block(X_test, test_ids, colons, optimizers, False, total_mean)

        if j == 0:
            print("a prediction 1: ", p1[0].data.cpu().numpy(), " ", "original image")
            print("a prediction 2: ", p2[0].data.cpu().numpy(), " ", augments[aug_ids[1]])
            print("a prediction 3: ", p3[0].data.cpu().numpy(), " ", augments[aug_ids[2]])
            print("a prediction 4: ", p4[0].data.cpu().numpy(), " ", augments[aug_ids[3]])
            print("a prediction 5: ", p5[0].data.cpu().numpy(), " ", augments[aug_ids[4]])
            print("a prediction 6: ", p6[0].data.cpu().numpy(), " ", augments[aug_ids[5]])

            print()

            print("a prediction 1: ", p1[20].data.cpu().numpy(), " ", "original image")
            print("a prediction 2: ", p2[20].data.cpu().numpy(), " ", augments[aug_ids[1]])
            print("a prediction 3: ", p3[20].data.cpu().numpy(), " ", augments[aug_ids[2]])
            print("a prediction 4: ", p4[20].data.cpu().numpy(), " ", augments[aug_ids[3]])
            print("a prediction 5: ", p5[20].data.cpu().numpy(), " ", augments[aug_ids[4]])
            print("a prediction 6: ", p6[20].data.cpu().numpy(), " ", augments[aug_ids[5]])

        avg_loss += mim.item()
        for i in range(p1.shape[0]):
            val, index = torch.max(p1[i], 0)
            val, index2 = torch.max(p2[i], 0)
            val, index3 = torch.max(p3[i], 0)
            val, index4 = torch.max(p4[i], 0)
            val, index5 = torch.max(p5[i], 0)
            val, index6 = torch.max(p6[i], 0)

            preds = [index.data.cpu().numpy(), index2.data.cpu().numpy(), index3.data.cpu().numpy(),
                     index4.data.cpu().numpy(), index5.data.cpu().numpy(), index6.data.cpu().numpy()]

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
              element,
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
    x_train_fileName = "..\\data\\stl10_binary\\train_X.bin"
    unlabeled_fileName = "..\\data\\stl10_binary\\unlabeled_X.bin"
    X_train = read_all_images(x_train_fileName)

    train_y_File = "..\\data\\stl10_binary\\train_y.bin"
    y_train = read_labels(train_y_File)

    testFile = "..\\data\\stl10_binary\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "..\\data\\stl10_binary\\test_y.bin"
    targets = read_labels(test_y_File)

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'encoder' + '.model'
    net_path = os.path.join(script_directory, filepath)

    encoder = UnsupervisedNet(1, INPUT_NET, NUMBER_CLASSES).to('cuda')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    max_loss = 1999
    max_loss_iter = 0
    most_clusters = 0
    most_clusters_iter = 0

    print(encoder)
    total_mean = torch.ones([NUMBER_CLASSES]) * 0.1
    total_mean = total_mean.to('cuda')

    labels_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    for idx, i in enumerate(y_train):
        labels_dict[i].append(idx)

    test_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    for idx, i in enumerate(targets):
        test_dict[i].append(idx)

    labels_to_imags = {1: "airplane",
                       2: "bird",
                       3: "car",
                       4: "cat",
                       5: "deer",
                       6: "dog",
                       7: "horse",
                       8: "monkey",
                       9: "ship",
                       10: "truck"}

    for iteration in range(MAX_STEPS_DEFAULT):
        # ids = []
        # samples_per_cluster = BATCH_SIZE_DEFAULT // 10
        #
        # for i in range(1, 11):
        #     ids += random.sample(labels_dict[i], samples_per_cluster)

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        p1, p2, p3, p4, p5, p6, mim, total_mean, orig_image, aug_ids = forward_block(X_train, ids, encoder, optimizer, train, total_mean)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            print("iteration: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best loss: ", max_loss_iter,
                  ": ", max_loss)
            print("description: ", DESCRIPTION)

            # test_ids = []
            #
            # for i in range(1, 11):
            #     test_ids += random.sample(test_dict[i], samples_per_cluster)

            test_ids = np.random.choice(len(X_test), size=BATCH_SIZE_DEFAULT, replace=False)
            print()
            p1, p2, p3, p4, p5, p6, mim, total_mean, orig_image, aug_ids = forward_block(X_test, test_ids, encoder, optimizer, False, total_mean)

            image_dict = print_info(p1, p2, p3, p4, p5, p6, targets, test_ids)

            loss, clusters = measure_acc_augments(X_test, encoder, targets, total_mean)

            if clusters >= MIN_CLUSTERS_TO_SAVE:
                for i in image_dict.keys():
                    numpy_cluster = torch.zeros([len(image_dict[i]), 1, SIZE, SIZE])
                    counter = 0
                    for index in image_dict[i]:
                        numpy_cluster[counter] = orig_image.cpu().detach()[index]
                        counter += 1
                    if counter > 0:
                        save_cluster(numpy_cluster, i, iteration)

                for i in image_dict.keys():
                    for index in image_dict[i]:
                        save_image(orig_image.cpu().detach()[index], index, "iter_"+str(iteration)+"_"+labels_to_imags[targets[test_ids[index]]], i)

            if clusters >= most_clusters:
                most_clusters = clusters
                most_clusters_iter = iteration

                if max_loss > loss:
                    max_loss = loss
                    max_loss_iter = iteration

                    print("models saved iter: " + str(iteration))
                    torch.save(encoder, net_path)

            print("most clusters: " + str(most_clusters) + " at iter: " + str(most_clusters_iter))


def to_tensor(X):
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


def print_info(p1, p2, p3, p4, p5, p6, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}
    image_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    counter = 0
    for i in range(p1.shape[0]):
        val, index = torch.max(p1[i], 0)
        val, index2 = torch.max(p2[i], 0)
        val, index3 = torch.max(p3[i], 0)
        val, index4 = torch.max(p4[i], 0)
        val, index5 = torch.max(p5[i], 0)
        val, index6 = torch.max(p6[i], 0)

        verdict = most_frequent([int(index.data.cpu().numpy()), int(index2.data.cpu().numpy()), int(index3.data.cpu().numpy()), int(index4.data.cpu().numpy()),
                                int(index5.data.cpu().numpy()), int(index6.data.cpu().numpy())])

        string = str(index.data.cpu().numpy()) + " " + str(index2.data.cpu().numpy()) + " " + str(
            index3.data.cpu().numpy()) + " " + str(index4.data.cpu().numpy()) + " " + str(index5.data.cpu().numpy()) + " " + str(index6.data.cpu().numpy()) + ", "

        image_dict[verdict].append(counter)
        counter += 1
        label = targets[test_ids[i]]
        if label == 10:
            label = 0
        print_dict[label] += string

    for i in print_dict.keys():
        print(i, " : ", print_dict[i])

    for i in image_dict.keys():
        print(i, " : ", image_dict[i])

    return image_dict


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