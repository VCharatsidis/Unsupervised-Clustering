from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from stl10_input import read_all_images, read_labels

import sys
from stl_utils import *

from torchvision.utils import make_grid
import matplotlib
from linear_net import LinearNet
from RandomNet import RandomNet
from SupervisedClassifier import SupervisedClassifier

# Default constants

LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 300000

BATCH_SIZE_DEFAULT = 200
INPUT_NET = 128
SIZE = 44
NETS = 1
EVAL_FREQ_DEFAULT = 50
PATIENCE = 20

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
FLAGS = None


############ UNSUPERVISED INFO #############################
classes_encoder = 20

conv_layers_encoder = 4
number_filters = 512
linear_layers_encoder = 1

batch_size = 100
lr = 1e-4

augments_compared = 3
heads = 5

encoder_name = "best_loss"
encoder_name = "most_clusters"
encoder = torch.load(encoder_name+"_encoder.model")

# encoder_name = "one_net_best_loss.model"
# encoder = torch.load(encoder_name)

encoder.eval()

# random_net = "random_encoder.model"
# encoder = RandomNet().to('cuda')
# torch.save(encoder, random_net)
# encoder = torch.load(random_net)
# encoder.eval()


DESCRIPTION = ["RANDOM NET"]

DESCRIPTION = ["RANDOM NET: ", " Image size: " + str(SIZE)\
              +",  BATCH SIZE: " + str(batch_size)\
              +",  lr: " + str(lr) + ",  train iters: 75000"
              ,",  Classes: " + str(classes_encoder)\
              ,",  embedding dim: " + str(INPUT_NET)\
              +",  conv layers: " + str(conv_layers_encoder)\
              +",  linear layers: " + str(linear_layers_encoder)\
              ,",  number filters: " + str(number_filters)\
              ,",  augments compared: " + str(augments_compared)\
              +",  heads: " + str(heads)\
              +",  Augments policy: only rc and then sobels_x_y. push loss n=10 " + "  " + encoder_name]


def encode(image):
    pad = (96 - SIZE) // 2
    original_image = scale(image, SIZE, pad, BATCH_SIZE_DEFAULT)
    original_image = original_image[:, :, pad:96 - pad, pad:96 - pad]

    #show_gray(original_image)

    original_image = sobel_filter_y(original_image, BATCH_SIZE_DEFAULT)
    original_image = original_image.to('cuda')

    encoding, _, _, _, _, _ = encoder(original_image)
    #encoding = encode_one_net(image)

    return encoding


def encode_one_net(image):
    pad = (96 - SIZE) // 2
    image /= 255

    crop_pad = 22
    crop_preparation = scale(image, 52, crop_pad, BATCH_SIZE_DEFAULT)
    crop_preparation = crop_preparation[:, :, crop_pad:96 - crop_pad, crop_pad:96 - crop_pad]
    crop_prep_horizontal = horizontal_flip(crop_preparation)

    soft_bin = binary(image)
    soft_bin = scale(soft_bin, SIZE, pad, BATCH_SIZE_DEFAULT)

    soft_bin = soft_bin[:, :, pad:96 - pad, pad:96 - pad]
    rev_soft_bin = torch.abs(1 - soft_bin)

    horiz_f = horizontal_flip(image, BATCH_SIZE_DEFAULT)
    soft_bin_hf = binary(horiz_f)
    soft_bin_hf = scale(soft_bin_hf, SIZE, pad, BATCH_SIZE_DEFAULT)
    soft_bin_hf = soft_bin_hf[:, :, pad:96 - pad, pad:96 - pad]

    rev_soft_bin_hf = torch.abs(1 - soft_bin_hf)

    # image = sobel_total(image, BATCH_SIZE_DEFAULT)
    rot = rotate(image, 20, BATCH_SIZE_DEFAULT)
    scale_rot = scale(rot, SIZE, pad, BATCH_SIZE_DEFAULT)
    scale_rot = scale_rot[:, :, pad:96 - pad, pad:96 - pad]

    rev_rot = rotate(image, -20, BATCH_SIZE_DEFAULT)
    scale_rev_rot = scale(rev_rot, SIZE, pad, BATCH_SIZE_DEFAULT)
    scale_rev_rot = scale_rev_rot[:, :, pad:96 - pad, pad:96 - pad]

    original_image = scale(image, SIZE, pad, BATCH_SIZE_DEFAULT)
    original_image = original_image[:, :, pad:96 - pad, pad:96 - pad]

    augments = {0: horizontal_flip(original_image, BATCH_SIZE_DEFAULT),

                1: scale(original_image, SIZE - 8, 4, BATCH_SIZE_DEFAULT),
                2: scale_rot,
                3: scale_rev_rot,
                4: random_erease(original_image, BATCH_SIZE_DEFAULT),
                5: sobel_filter_x(original_image, BATCH_SIZE_DEFAULT),
                6: sobel_filter_y(original_image, BATCH_SIZE_DEFAULT),
                7: sobel_total(original_image, BATCH_SIZE_DEFAULT),
                8: sobel_filter_x(horizontal_flip(original_image, BATCH_SIZE_DEFAULT), BATCH_SIZE_DEFAULT),
                9: sobel_filter_y(horizontal_flip(original_image, BATCH_SIZE_DEFAULT), BATCH_SIZE_DEFAULT),
                10: sobel_total(horizontal_flip(original_image, BATCH_SIZE_DEFAULT), BATCH_SIZE_DEFAULT),
                # 12: binary(original_image),
                # 13: binary(horizontal_flip(original_image, BATCH_SIZE_DEFAULT)),
                # 14: binary(scale_rot),
                # 15: binary(scale_rev_rot),
                11: soft_bin,
                12: rev_soft_bin,
                13: soft_bin_hf,
                14: rev_soft_bin_hf,
                15: torch.abs(1 - original_image)
                # 16: center_crop(image, SIZE, BATCH_SIZE_DEFAULT),
                }

    ids = np.random.choice(len(augments), size=6, replace=False)

    p = torch.zeros([BATCH_SIZE_DEFAULT, INPUT_NET]).to('cuda')
    mean = torch.zeros([BATCH_SIZE_DEFAULT, classes_encoder])

    encoding1, _ = encoder(random_crop(crop_preparation, SIZE, BATCH_SIZE_DEFAULT).to('cuda'), p, p, p, p, p, mean)
    encoding2, _ = encoder(random_crop(crop_prep_horizontal, SIZE, BATCH_SIZE_DEFAULT).to('cuda'), p, p, p, p, p, mean)
    encoding3, _ = encoder(augments[ids[3]].to('cuda'), p, p, p, p, p, mean)
    encoding4, _ = encoder(augments[ids[4]].to('cuda'), p, p, p, p, p, mean)
    encoding5, _ = encoder(augments[ids[5]].to('cuda'), p, p, p, p, p, mean)

    encoding, _ = encoder(original_image.to('cuda'), encoding1, encoding2, encoding3, encoding4, encoding5, mean)
    return encoding


def forward_block(X, ids, classifier, optimizer, train, targets):
    images = X[ids, :]
    x_tensor = to_tensor(images)

    x_train = rgb2gray(x_tensor)
    x_tensor = to_tensor(x_train)
    x_tensor = x_tensor.unsqueeze(0)
    images = x_tensor.transpose(0, 1)
    images = images.transpose(2, 3)

    images = images / 255.0

    with torch.no_grad():
        encodings = encode(images)

    preds = classifier(encodings.detach())

    # pad = (96 - SIZE) // 2
    # original_image = scale(images, SIZE, pad, BATCH_SIZE_DEFAULT)
    # original_image = original_image[:, :, pad:96 - pad, pad:96 - pad]
    # original_image = original_image.to('cuda')
    # preds = classifier(original_image)

    tensor_targets = torch.LongTensor(targets[ids]).unsqueeze(dim=1).to('cuda')
    y_onehot = torch.FloatTensor(BATCH_SIZE_DEFAULT, 10).to('cuda')
    y_onehot.zero_()
    y_onehot.scatter_(1, tensor_targets, 1)

    cross_entropy_loss = - (y_onehot * torch.log(preds)).sum(dim=1).mean()

    if train:
        # for p in encoder.parameters():
        #     print(p.name, p.data)

        for p in encoder.parameters():
            p.requires_grad = False

        optimizer.zero_grad()
        cross_entropy_loss.backward(retain_graph=True)
        optimizer.step()

    return preds, cross_entropy_loss


def save_image(original_image, iteration, name):
    sample = original_image.view(-1, 1, original_image.shape[2], original_image.shape[2])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"gen_images/{name}_iter_{iteration}.png", sample)


def accuracy(predictions, targets):
   predictions = predictions.cpu().detach().numpy()
   preds = np.argmax(predictions, 1)
   result = preds == targets
   sum = np.sum(result)
   #accur = sum / float(targets.shape[0])

   return sum


def measure_acc_augments(X_test, classifier, targets):
    runs = len(X_test) // BATCH_SIZE_DEFAULT
    avg_loss = 0
    sum_correct = 0

    for j in range(runs):
        test_ids = range(j * BATCH_SIZE_DEFAULT, (j + 1) * BATCH_SIZE_DEFAULT)
        test_ids = np.array(test_ids)
        optimizer = []
        preds, mim = forward_block(X_test, test_ids, classifier, optimizer, False, targets)

        sum_correct += accuracy(preds, targets[test_ids])
        avg_loss += mim.item()

    average_test_loss = avg_loss / runs
    accuracy_test_set = sum_correct / (runs * BATCH_SIZE_DEFAULT)
    print("Test set avg loss: ", average_test_loss, " avg accuracy: ", accuracy_test_set)

    return average_test_loss, accuracy_test_set


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
    fileName = "..\\data\\stl10_binary\\train_X.bin"
    X_train = read_all_images(fileName)

    train_y_File = "..\\data\\stl10_binary\\train_y.bin"
    y_train = read_labels(train_y_File)

    testFile = "..\\data\\stl10_binary\\test_X.bin"
    X_test = read_all_images(testFile)

    test_y_File = "..\\data\\stl10_binary\\test_y.bin"
    targets = read_labels(test_y_File)

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'encoders\\encoder_' + str(0) + '.model'
    path_to_model = os.path.join(script_directory, filepath)

    linearClassifier = LinearNet(INPUT_NET).to('cuda')
    optimizer = torch.optim.Adam(linearClassifier.parameters(), lr=LEARNING_RATE_DEFAULT)

    # linearClassifier = SupervisedClassifier(INPUT_NET).to('cuda')
    # optimizer = torch.optim.Adam(linearClassifier.parameters(), lr=LEARNING_RATE_DEFAULT)

    best_accuracy = 0
    iter_acc = 0
    patience = 50

    for iteration in range(MAX_STEPS_DEFAULT):
        linearClassifier.train()
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        preds, mim = forward_block(X_train, ids, linearClassifier, optimizer, train, y_train)

        if iteration % EVAL_FREQ_DEFAULT == 0:
            linearClassifier.eval()
            print()
            print("==============================================================")
            print("ITERATION: ", iteration,
                  ", batch size: ", BATCH_SIZE_DEFAULT,
                  ", lr: ", LEARNING_RATE_DEFAULT,
                  ", best acc: ", iter_acc,
                  ": ", best_accuracy)
            print("patience: ", patience)
            print()
            for info in DESCRIPTION:
                print(info)
            print()
            loss, acc = measure_acc_augments(X_test, linearClassifier, targets)

            if best_accuracy < acc:
                best_accuracy = acc
                iter_acc = iteration
                patience = 0
                print("models saved iter: " + str(iteration))
            else:
                patience += 1

            if patience > PATIENCE:
                print("For ", patience, " iterations we do not have a better accuracy")
                print("best accuracy: ", best_accuracy, " at iter: ", iter_acc)
                print("accuracy at stop: ", acc, "loss at stop: ", loss)

                file = open("experiments.txt", "a")
                for info in DESCRIPTION:
                    file.write(info)

                accuracy_info = ",  BEST ACCURACY: " + str(best_accuracy) + " at iter: " + str(iter_acc) + "\n"
                file.write(accuracy_info)
                file.close()
                break;


def to_tensor(X):
    #X = np.reshape(X, (batch_size, 1, 96, 96))
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