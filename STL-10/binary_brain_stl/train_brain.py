from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from stl10_input import read_all_images, read_labels
from brain_stl import BinBrainSTL
from stl_utils import *
import random
import sys


EPS = sys.float_info.epsilon
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 400

INPUT_NET = 2048
EMBEDING_SIZE = 1024
SIZE = 32

NEGATIVES = BATCH_SIZE_DEFAULT
EVAL_FREQ_DEFAULT = 500

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)
FLAGS = None

ELEMENTS_ABOVE_DIAG = ((BATCH_SIZE_DEFAULT - 1) * (BATCH_SIZE_DEFAULT)) / 2

ranges = {}
for i in range(BATCH_SIZE_DEFAULT-1):
    ranges[i] = [(x + i + 1) % BATCH_SIZE_DEFAULT for x in range(BATCH_SIZE_DEFAULT)]

transformations_dict ={0: "original",
                       1: "scale",
                       2: "rotate",
                       3: "reverse pixel value",
                       4: "sobel total",
                       5: "sobel x",
                       6: "sobel y",
                       7: "gaussian blur",
                       8: "random crop 66",
                       9: "random crop 76",
                       10: "random_erease",
                       11: "random crop"}


def energy_loss(pred):
    vertical_mean = pred.mean(dim=1)
    reverse_vertical_mean = 1 - vertical_mean
    log_rev_vert_mean = - torch.log(reverse_vertical_mean)
    energy = log_rev_vert_mean.mean(dim=0)

    return energy


def binary_loss(preds_1, preds_2):
    '''
    The energy loss simply tries to keep alla elements of the predictions 0
    '''
    energy_loss_1 = energy_loss(preds_1)
    energy_loss_2 = energy_loss(preds_2)

    mean_energy_loss = (energy_loss_1 + energy_loss_2) / 2

    '''
    The 2 (can be more) transformations of the same image are attracted with the product mechanism
    '''
    product = preds_1 * preds_2

    '''
    The three lines below (batch vertical mean)
    push to have a value in every dimension of the representation. 
    So that there are not obsolete dimensions never used from one representation.
    '''
    product_vertical_mean = product.mean(dim=0)
    log_vertical_mean = - torch.log(product_vertical_mean+EPS)
    batch_vertical_mean = log_vertical_mean.mean(dim=0)

    '''
    every product representation 
    which is the combined representation of both transformations
    of the same image, repels every other product representations
    which comes from different images. As a matter of fact the loss goes
    down only if the transfromations of the same image attract each other while
    creating a preduct that not alligns with products of other images.
    '''

    rev_prod = 1 - product

    transposed = rev_prod.transpose(0, 1)
    nondiag = torch.mm(product, transposed)
    nondiag = nondiag / EMBEDING_SIZE

    log_nondiag = - torch.log(nondiag + EPS)
    cleaned = torch.triu(log_nondiag, diagonal=1)

    negative_repel = cleaned.sum(dim=0).sum(dim=0) / ELEMENTS_ABOVE_DIAG

    '''
    Finally, we combine the three losses above with whatever coefficients we want.
    Batch vertical mean might not be needed at all.
    '''

    total_loss = 0.9 * mean_energy_loss + 0.07 * negative_repel + 0.03 * batch_vertical_mean

    return total_loss


def forward_block(X, ids, encoder, optimizer, train):
    number_transforms = len(transformations_dict.keys())
    aug_ids = np.random.choice(number_transforms, size=number_transforms, replace=False)

    image = X[ids, :]

    if train:
        image = rgb2gray(image)

        image = to_tensor(image)
        image = image.unsqueeze(0)
        image = image.transpose(0, 1)
        image = image.transpose(2, 3)

        image /= 255

    fourth = BATCH_SIZE_DEFAULT // 4

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

    # print(transformations_dict[aug_ids[2]])
    # show_gray(image_3)
    #
    # print(transformations_dict[aug_ids[3]])
    # show_gray(image_4)
    #
    # print(transformations_dict[aug_ids[4]])
    # show_gray(image_5)
    #
    # print(transformations_dict[aug_ids[5]])
    # show_gray(image_6)
    #
    # print(transformations_dict[aug_ids[6]])
    # show_gray(image_7)
    #
    # print(transformations_dict[aug_ids[7]])
    # show_gray(image_8)

    _, _, test_preds_1 = encoder(image_1.to('cuda'))
    _, _, test_preds_2 = encoder(image_2.to('cuda'))

    test_total_loss = binary_loss(test_preds_1, test_preds_2)

    if train:
        optimizer.zero_grad()
        test_total_loss.backward()
        optimizer.step()

    return test_preds_1, test_preds_2, test_total_loss


def transformation(id, image):
    pad = (96 - SIZE) // 2
    fourth = BATCH_SIZE_DEFAULT // 4

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


def measure_acc_augments(x_test, encoder):
    size = BATCH_SIZE_DEFAULT
    runs = len(x_test) // size
    sum_loss = 0

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)

        with torch.no_grad():
            test_preds_1, test_preds_2, test_total_loss = forward_block(x_test, test_ids, encoder, [], False)

        sum_loss += test_total_loss.item()

    avg_loss = sum_loss / runs

    print()
    print("Avg test loss: ", avg_loss)
    print()

    return avg_loss


def print_params(model):
    for param in model.parameters():
        print(param.data)


def preprocess_stl(x):
    images = rgb2gray(x)

    images = to_tensor(images)
    images = images.unsqueeze(0)
    images = images.transpose(0, 1)
    images = images.transpose(2, 3)

    images /= 255

    return images


def train():
    validation_path = "..\\data\\stl10_binary\\train_X.bin"
    X_validation = read_all_images(validation_path)
    X_validation = preprocess_stl(X_validation)

    unsupervised_path = "..\\data_2\\stl10_binary\\unlabeled_X.bin"
    X_train = read_all_images(unsupervised_path)
    # X_train = preprocess_stl(X_train)

    # test_path = "..\\data\\stl10_binary\\test_X.bin"
    # X_test = read_all_images(test_path)
    # X_test = preprocess_stl(X_test)

    ##############################################

    train_y_path = "..\\data\\stl10_binary\\train_y.bin"
    y_train = read_labels(train_y_path)
    y_validation = np.array([x % 10 for x in y_train])

    # test_y_path = "..\\data\\stl10_binary\\test_y.bin"
    # y_test = read_labels(test_y_path)
    # y_test = np.array([x % 10 for x in y_test])

    ###############################################

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'stl_binary' + '.model'
    clusters_net_path = os.path.join(script_directory, filepath)

    encoder = BinBrainSTL(1, EMBEDING_SIZE).cuda()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE_DEFAULT)

    max_loss_iter = 0
    test_best_loss = 1000

    print(encoder)
    print("X_train: ", X_train.shape, " X_validation: ", X_validation.shape, " targets: ", y_validation.shape)

    for iteration in range(MAX_STEPS_DEFAULT):
        encoder.train()

        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        test_preds_1, test_preds_2, test_total_loss = forward_block(X_train, ids, encoder, optimizer, train)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            encoder.eval()
            print("==================================================================================")

            test_loss = measure_acc_augments(X_validation, encoder)

            if test_loss < test_best_loss:
                test_best_loss = test_loss
                max_loss_iter = iteration
                print("models saved iter: " + str(iteration))
                torch.save(encoder, clusters_net_path)

            print("ITERATION: ", iteration,
                  ",  batch size: ", BATCH_SIZE_DEFAULT,
                  ",  lr: ", LEARNING_RATE_DEFAULT,
                  ",  best loss iter: ", max_loss_iter,
                  ",  negatives: ", NEGATIVES)



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