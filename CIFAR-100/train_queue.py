from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.utils.prune as prune

import sys

import argparse
import os

from binary_net import DeepBinBrainCifar

from stl_utils import *
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

BATCH_SIZE_DEFAULT = 512

EMBEDINGS = 512
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
ZERO_DIAG = square.fill_diagonal_(0)

ELEMENTS_EXCEPT_DIAG = BATCH_SIZE_DEFAULT * (BATCH_SIZE_DEFAULT - 1)

first = True

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
        return rotate(color_jitter(image), 46)
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
        rot = rotate(scaled_up, 46)
        return rot

    elif id == 12:
        t = random_crop(color_jitter(image), 22, quarter, 22)
        scaled_up = scale_up(t, 32, 32, quarter)
        blured = gaussian_blur(scaled_up)
        rot = rotate(blured, 46)
        return rot

    elif id == 13:
        t = random_crop(color_jitter(image), 22, quarter, 22)
        scaled_up = scale_up(t, 32, 32, quarter)

        return scaled_up

    elif id == 14:
        t = random_crop(color_jitter(image), 22, quarter, 22)
        scaled_up = scale_up(t, 32, 32, quarter)
        return scaled_up

    elif id == 15:
        t = random_crop(color_jitter(image), 22, quarter, 22)
        scaled_up = scale_up(t, 32, 32, quarter)
        sobeled = sobel_filter_x(scaled_up, quarter)

        AA = sobeled.view(sobeled.size(0), -1)
        AA -= AA.min(1, keepdim=True)[0]
        AA /= AA.max(1, keepdim=True)[0]
        AA = AA.view(quarter, 1, SIZE, SIZE)
        return AA

    elif id == 16:
        t = random_crop(color_jitter(image), 26, quarter, 26)
        scaled_up = scale_up(t, 32, 32, quarter)
        return scaled_up

    print("Error in transformation of the image.")
    return image


def new_agreement(product, denominator, rev_prod):
    transposed = rev_prod.transpose(0, 1)

    nondiag = torch.mm(product, transposed)
    nondiag = nondiag / denominator.unsqueeze(dim=1)

    log_nondiag = - torch.log(nondiag)
    cleaned = log_nondiag * ZERO_DIAG.cuda()

    negative = cleaned.sum(dim=0).sum(dim=0) / ELEMENTS_EXCEPT_DIAG

    return negative


def queue_agreement(product, denominator, rev_prod):
    transposed = rev_prod.transpose(0, 1)

    nondiag = torch.mm(product, transposed)
    nondiag = nondiag / denominator.unsqueeze(dim=1)

    log_nondiag = - torch.log(nondiag)
    negative = log_nondiag.mean()

    return negative


def forward_block(X, ids, encoder, optimizer, train, rev_product):
    global first
    number_transforms = 16
    aug_ids = np.random.choice(number_transforms, size=number_transforms, replace=False)

    image = X[ids, :]

    fourth = BATCH_SIZE_DEFAULT // 4
    image_1 = transformation(aug_ids[0], image[0:fourth])
    image_2 = transformation(aug_ids[1], image[0:fourth])
    # image_3 = transformation(aug_ids[2], image[0:fourth])
    # image_4 = transformation(aug_ids[3], image[0:fourth])

    image_5 = transformation(aug_ids[4], image[fourth: 2 * fourth])
    image_6 = transformation(aug_ids[5], image[fourth: 2 * fourth])
    # image_7 = transformation(aug_ids[6], image[fourth: 2 * fourth])
    # image_8 = transformation(aug_ids[7], image[fourth: 2 * fourth])

    image_9 = transformation(aug_ids[8], image[2 * fourth: 3 * fourth])
    image_10 = transformation(aug_ids[9], image[2 * fourth: 3 * fourth])
    # image_11 = transformation(aug_ids[10], image[2 * fourth: 3 * fourth])
    # image_12 = transformation(aug_ids[11], image[2 * fourth: 3 * fourth])

    image_13 = transformation(aug_ids[12], image[3 * fourth:])
    image_14 = transformation(aug_ids[13], image[3 * fourth:])
    # image_15 = transformation(aug_ids[14], image[3 * fourth:])
    # image_16 = transformation(aug_ids[15], image[3 * fourth:])

    # save_images(image_1, aug_ids[0])
    # save_images(image_2, aug_ids[1])
    # save_images(image_3, aug_ids[2])
    # save_images(image_4, aug_ids[3])
    # save_images(image_5, aug_ids[4])
    # save_images(image_6, aug_ids[5])
    # save_images(image_7, aug_ids[6])
    # save_images(image_8, aug_ids[7])

    image_1 = torch.cat([image_1, image_5, image_9, image_13], dim=0)
    image_2 = torch.cat([image_2, image_6, image_10, image_14], dim=0)
    # image_3 = torch.cat([image_3, image_7, image_11, image_15], dim=0)
    # image_4 = torch.cat([image_4, image_8, image_12, image_16], dim=0)

    # save_images(image_1, 12)
    # save_images(image_2, 13)

    #print(list(encoder.brain[0].weight))

    _, _, probs10_a = encoder(image_1.to('cuda'))
    _, _, probs10_b = encoder(image_2.to('cuda'))
    # _, _, probs10_c = encoder(image_3.to('cuda'))
    # _, _, probs10_d = encoder(image_4.to('cuda'))

    product = probs10_a * probs10_b  # * probs10_c * probs10_d
    current_rev = 1 - product
    denominator = probs10_a.sum(dim=1) + probs10_b.sum(dim=1) #+ probs10_c.sum(dim=1) + probs10_d.sum(dim=1)

    new_loss = new_agreement(product, denominator, current_rev)

    if first or not train:
        # print("first: ", first)
        total_loss = new_loss
        rev_product = current_rev.detach()
        first = False

    else:
        # print("first", first)
        old_loss = queue_agreement(product, denominator, rev_product)
        rev_product = torch.cat([rev_product, current_rev.detach()])
        total_loss = new_loss + old_loss

    if train:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return probs10_a, probs10_b, total_loss, rev_product


def save_cluster(original_image, cluster, iteration):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"iter_{iteration}_c_{cluster}.png", sample)


def save_image(original_image, index, name, cluster=0):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[3])
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


def preproccess_cifar(x):
    x = to_tensor(x)

    x = x.transpose(1, 3)

    # pad = (SIZE-SIZE_Y) // 2
    # x = x[:, :, :, pad: SIZE - pad]
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

    filepath = 'cifar100_models\\queue' + '.model'
    clusters_net_path = os.path.join(script_directory, filepath)

    encoder = DeepBinBrainCifar(1, EMBEDINGS).to('cuda')

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

            if iteration >= 50:
                rev_product = rev_product[BATCH_SIZE_DEFAULT:, :]

        print("==================================================================================")
        print("batch mean ones: ",
              (np.where(probs10.data.cpu().numpy() > 0.5))[0].shape[0] / (probs10[0].shape[0] * BATCH_SIZE_DEFAULT))

        print("train avg loss : ", avg_loss / EVAL_FREQ_DEFAULT)
        avg_loss = 0
        encoder.eval()

        test_loss = measure_acc_augments(X_test, encoder, rev_product)

        if test_loss < test_best_loss:
            test_best_loss = test_loss
            max_loss_iter = total_iters
            min_miss_percentage = test_loss
            print("models saved iter: " + str(total_iters))
            torch.save(encoder, clusters_net_path)

        print("EPOCH: ", epoch,
              "Total ITERATION: ", total_iters,
              " epoch iter: ", iteration,
              ",  batch size: ", BATCH_SIZE_DEFAULT,
              ",  lr: ", LEARNING_RATE_DEFAULT,
              ",  best loss iter: ", max_loss_iter,
              "-", min_miss_percentage
              , DESCRIPTION)


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