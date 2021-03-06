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

BATCH_SIZE_DEFAULT = 24

INPUT_NET = 4608
SIZE = 32
NETS = 1
UNSUPERVISED = False
#DROPOUT = [0.90, 0.90, 0.90, 0.90, 0.90]
DROPOUT = 0
class_n = 10
CLASSES = [class_n, class_n, class_n, class_n, class_n]
DESCRIPTION = " Image size: "+str(SIZE) + " , Dropout2d: "+str(DROPOUT)+" , Classes: "+str(CLASSES)

EVAL_FREQ_DEFAULT = 500
MIN_CLUSTERS_TO_SAVE = 10
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
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
    pad = (96 - SIZE) // 2
    image /= 255

    crop_size = 56
    crop_pad = (96 - crop_size) // 2
    crop_preparation = scale(image, crop_size, crop_pad, BATCH_SIZE_DEFAULT)
    crop_preparation = crop_preparation[:, :, crop_pad:96 - crop_pad, crop_pad:96 - crop_pad]
    # crop_prep_horizontal = horizontal_flip(crop_preparation)

    crop_size2 = 66
    crop_pad2 = (96 - crop_size2) // 2
    crop_preparation2 = scale(image, crop_size2, crop_pad2, BATCH_SIZE_DEFAULT)
    crop_preparation2 = crop_preparation2[:, :, crop_pad2:96 - crop_pad2, crop_pad2:96 - crop_pad2]
    crop_prep_horizontal2 = horizontal_flip(crop_preparation2)

    horiz_f = horizontal_flip(image, BATCH_SIZE_DEFAULT)

    soft_bin_hf = binary(horiz_f)
    soft_bin_hf = scale(soft_bin_hf, SIZE, pad, BATCH_SIZE_DEFAULT)
    soft_bin_hf = soft_bin_hf[:, :, pad:96 - pad, pad:96 - pad]
    rev_soft_bin_hf = torch.abs(1 - soft_bin_hf)
    rev_soft_bin_hf /= 255

    original_image = scale(image, SIZE, pad, BATCH_SIZE_DEFAULT)
    original_image = original_image[:, :, pad:96 - pad, pad:96 - pad]

    original_hfliped = color_jitter(horizontal_flip(original_image, BATCH_SIZE_DEFAULT))

    augments = {0: original_hfliped,
                1: scale(original_hfliped, SIZE-8, 4, BATCH_SIZE_DEFAULT),
                2: random_erease(original_hfliped, BATCH_SIZE_DEFAULT),
                3: sobel_filter_x(original_hfliped, BATCH_SIZE_DEFAULT),
                4: sobel_filter_y(original_hfliped, BATCH_SIZE_DEFAULT),
                5: sobel_total(original_hfliped, BATCH_SIZE_DEFAULT),
                6: soft_bin_hf,
                7: rev_soft_bin_hf,
                8: torch.abs(1 - original_hfliped),
                9: rotate(original_hfliped, 40),
                10: scale(original_hfliped, SIZE - 12, 6, BATCH_SIZE_DEFAULT),
                }

    ids = np.random.choice(len(augments), size=len(augments.keys()), replace=False)

    image_1 = color_jitter(random_crop(crop_preparation, SIZE, BATCH_SIZE_DEFAULT))
    image_2 = color_jitter(random_crop(crop_prep_horizontal2, SIZE, BATCH_SIZE_DEFAULT))
    image_3 = augments[ids[2]]
    image_4 = color_jitter(original_image)

    ####################### print augments #############################################
    # for id in ids:
    #     numpy_cluster = torch.zeros([BATCH_SIZE_DEFAULT, 1, SIZE, SIZE])
    #     image_3 = augments[id]
    #
    #     # step 1: convert it to [0 ,2]
    #     image_3 = image_3 + 1
    #
    #     # step 2: convert it to [0 ,1]
    #     image_3 = image_3 - image_3.min()
    #     image_3 = image_3 / (image_3.max() - image_3.min())
    #
    #     counter = 0
    #     for im in image_3:
    #         numpy_cluster[counter] = im.cpu().detach()
    #         counter += 1
    #     print("max",  torch.max(image_3))
    #     print("min", torch.min(image_3))
    #     print(id)
    #     save_cluster(numpy_cluster, id, 0)
    #
    # numpy_cluster = torch.zeros([BATCH_SIZE_DEFAULT, 1, SIZE, SIZE])
    # counter = 0
    # for im in image_1:
    #     numpy_cluster[counter] = im.cpu().detach()
    #     counter += 1
    # print("max", torch.max(image_3))
    # print("min", torch.min(image_3))
    # print(id)
    # save_cluster(numpy_cluster, 1, 1)
    #
    # numpy_cluster = torch.zeros([BATCH_SIZE_DEFAULT, 1, SIZE, SIZE])
    # counter = 0
    # for im in image_2:
    #     numpy_cluster[counter] = im.cpu().detach()
    #     counter += 1
    # print("max", torch.max(image_3))
    # print("min", torch.min(image_3))
    # print(id)
    # save_cluster(numpy_cluster, 2, 2)
    #
    # numpy_cluster = torch.zeros([BATCH_SIZE_DEFAULT, 1, SIZE, SIZE])
    # counter = 0
    # for im in image_4:
    #     numpy_cluster[counter] = im.cpu().detach()
    #     counter += 1
    # print("max", torch.max(image_3))
    # print("min", torch.min(image_3))
    # print(id)
    # save_cluster(numpy_cluster, 4, 4)

    show_gray(image_1)
    show_gray(image_2)
    show_gray(image_3)
    show_gray(image_4)

    _, test_preds_1, help_preds_1_1, help_preds_1_2,  help_preds_1_3, help_preds_1_4 = encoder(image_1.to('cuda'))
    _, test_preds_2, help_preds_2_1, help_preds_2_2,  help_preds_2_3, help_preds_2_4 = encoder(image_2.to('cuda'))
    _, test_preds_3, help_preds_3_1, help_preds_3_2,  help_preds_3_3, help_preds_3_4 = encoder(image_3.to('cuda'))
    _, test_preds_4, help_preds_4_1, help_preds_4_2, help_preds_4_3, help_preds_4_4 = encoder(image_4.to('cuda'))

    return test_preds_1, test_preds_2, test_preds_3, test_preds_4,\
           help_preds_1_1, help_preds_2_1,  help_preds_3_1, help_preds_4_1,\
           help_preds_1_2, help_preds_2_2, help_preds_3_2, help_preds_4_2,\
           help_preds_1_3, help_preds_2_3, help_preds_3_3, help_preds_4_3,\
           help_preds_1_4, help_preds_2_4, help_preds_3_4, help_preds_4_4,\
           original_image, ids


def entropy_minmax_loss(targets, preds_1, preds_2, preds_3, preds_4):
    # batch_cross_entropy_1 = batch_entropy(preds_1, targets)
    # batch_cross_entropy_2 = batch_entropy(preds_2, targets)
    # batch_cross_entropy_3 = batch_entropy(preds_3, targets)
    #
    # total_batch_cross_entropy = batch_cross_entropy_1 + batch_cross_entropy_2 + batch_cross_entropy_3

    product = preds_1 * preds_2 * preds_3 * preds_4
    product = product.mean(dim=0)
    log_product = torch.log(product)
    class_mean = - log_product.mean(dim=0)

    total_loss = class_mean #- total_batch_cross_entropy

    return total_loss


def forward_block(X, ids, encoder, optimizer, train, total_mean):
    x_train = X[ids, :]
    x_train = rgb2gray(x_train)

    x_tensor = to_tensor(x_train)
    x_tensor = x_tensor.unsqueeze(0)
    images = x_tensor.transpose(0, 1)
    images = images.transpose(2, 3)

    test_preds_1, test_preds_2, test_preds_3, test_preds_4, \
    help_preds_1_1, help_preds_2_1, help_preds_3_1, help_preds_4_1, \
    help_preds_1_2, help_preds_2_2, help_preds_3_2, help_preds_4_2, \
    help_preds_1_3, help_preds_2_3, help_preds_3_3, help_preds_4_3, \
    help_preds_1_4, help_preds_2_4, help_preds_3_4, help_preds_4_4, \
    orig_image, aug_ids = encode_4_patches(images, encoder)

    test_total_loss = entropy_minmax_loss(get_targets(CLASSES[0]), test_preds_1, test_preds_2, test_preds_3, test_preds_4)
    help_total_loss_1 = entropy_minmax_loss(get_targets(CLASSES[1]), help_preds_1_1, help_preds_2_1, help_preds_3_1, help_preds_4_1)
    help_total_loss_2 = entropy_minmax_loss(get_targets(CLASSES[2]), help_preds_1_2, help_preds_2_2, help_preds_3_2, help_preds_4_2)
    help_total_loss_3 = entropy_minmax_loss(get_targets(CLASSES[3]), help_preds_1_3, help_preds_2_3, help_preds_3_3, help_preds_4_3)
    help_total_loss_4 = entropy_minmax_loss(get_targets(CLASSES[4]), help_preds_1_4, help_preds_2_4, help_preds_3_4, help_preds_4_4)

    m_preds = (test_preds_1 + test_preds_2 + test_preds_3 + test_preds_4) / 4
    total_mean = 0.99 * total_mean + 0.01 * m_preds.mean(dim=0).detach()

    all_losses = test_total_loss + help_total_loss_1 + help_total_loss_2 + help_total_loss_3 + help_total_loss_4

    if train:
        optimizer.zero_grad()
        all_losses.backward(retain_graph=True)
        optimizer.step()

    return test_preds_1, test_preds_2, test_preds_3, test_preds_4, test_total_loss, total_mean, orig_image, aug_ids


def batch_entropy(pred, targets):
    batch_mean_preds = pred.mean(dim=0)
    H_batch = torch.log(batch_mean_preds).mean()

    return H_batch


def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


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
    size = BATCH_SIZE_DEFAULT
    runs = len(X_test)//size
    avg_loss = 0

    print()
    print("total mean:     ", total_mean.data.cpu().numpy())
    print()

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)
        optimizers = []
        p1, p2, p3, p4, mim, total_mean, orig_image, aug_ids = forward_block(X_test, test_ids, colons, optimizers, False, total_mean)

        avg_loss += mim.item()
        for i in range(p1.shape[0]):
            val, index = torch.max(p1[i], 0)
            val, index2 = torch.max(p2[i], 0)
            val, index3 = torch.max(p3[i], 0)
            val, index4 = torch.max(p4[i], 0)

            preds = [index.data.cpu().numpy(), index2.data.cpu().numpy(), index3.data.cpu().numpy(), index4.data.cpu().numpy()]

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

    filepath = 'models\\most_clusters_encoder' + '.model'
    clusters_net_path = os.path.join(script_directory, filepath)

    filepath = 'models\\best_loss_encoder' + '.model'
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
        p1, p2, p3, p4, mim, total_mean, orig_image, aug_ids = forward_block(X_train, ids, encoder, optimizer, train, total_mean)

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

            p1, p2, p3, p4, mim, total_mean, orig_image, aug_ids = forward_block(X_test, test_ids, encoder, optimizer, False, total_mean)
            classes_dict, numbers_classes_dict = print_info(p1, p2, p3, p4, targets, test_ids)

            loss, clusters = measure_acc_augments(X_test, encoder, targets, total_mean)

            if clusters >= MIN_CLUSTERS_TO_SAVE and max_loss > loss:
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


def print_info(p1, p2, p3, p4, targets, test_ids):
    print_dict = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "", 9: "", 0: ""}
    #image_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}
    image_dict = {x: [] for x in range(CLASSES[0])}
    numbers_classes_dict = {x: [] for x in range(CLASSES[0])}

    cluster = BATCH_SIZE_DEFAULT // 10
    counter_cluster =0
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