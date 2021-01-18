
from stl_utils import *
import cifar10_utils
import matplotlib
from torchvision.utils import make_grid
import sys
import pickle
import argparse
import os

from binary_net import DeepBinBrainCifar

from stl_utils import *

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(sci_mode=False)

SAVE_IMAGES = False
COARSE_LABELS = True
EMBEDDINGS = 64
#name = f"cifar100_models\\self_contrast_{EMBEDDINGS}_b2_0.model"
name = "cifar100_models\\supervised_queue_bonus5_half_3.model"
encoder = torch.load(name)
print("encoder: ", name)
encoder.eval()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

with open('data\\train', 'rb') as fo:
    res = pickle.load(fo, encoding='bytes')

meta = unpickle('data\\meta')

fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]

train = unpickle('data\\train')

filenames = [t.decode('utf8') for t in train[b'filenames']]


train_data = train[b'data']

test = unpickle('data\\test')

filenames = [t.decode('utf8') for t in test[b'filenames']]

if COARSE_LABELS:
    targets = test[b'coarse_labels']
else:
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

if COARSE_LABELS:
    PA = 500
else:
    PA = 100


def get_binaries(X_test):
    size = 500
    runs = len(X_test) // size

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)
        test_ids = np.array(test_ids)

        image = X_test[test_ids]

        with torch.no_grad():
            _, _, binary = encoder(image.cuda())

            binary = torch.round(binary)  # round the prediction to make it real binary.

            if j == 0:
                binaries = binary
            else:
                binaries = torch.cat([binaries, binary], dim=0)

    print("Binaries shape: ", binaries.shape)
    return binaries


ids = np.array(range(X_test.shape[0]))


def get_hamming(idx, binaries):
    '''
    Gets the index of 1 image and XOR it's binary with all other binaries of the test-set
    :param idx: The index of the compared image
    :param binaries: the binaries.
    :return: array with hammings and indexes sorted for the given image vs the hole test-set.
    '''

    with torch.no_grad():
        hammings = torch.logical_xor(binaries[idx], binaries[ids]).sum(dim=1)

    sorted, indices = torch.sort(hammings)

    if SAVE_IMAGES:
        save_image(X_test[indices[:16]], idx)

    s = [(indices[i].item(), sorted[i].item()) for i in range(PA)]  # make it a list and keep only the first top PA
    s = s[1:]  # Delete the first element which is going to be the compared image itself.

    return s


def average_precision(hammings, class_a):
    corrects = 0
    sum_precisions = 0
    for h in range(len(hammings)):
        index = hammings[h][0]
        c = targets[index]

        if class_a == c:
            corrects += 1

        precision = corrects / (h + 1)
        sum_precisions += precision

    ap = sum_precisions / len(hammings)
    recall = corrects / len(hammings)
    return ap, recall


def get_mAP_and_Top1(binaries):
    sum_top1 = 0
    sum_top500 = 0
    sum_bottom1000 = 0
    sum_av_precision = 0

    sum_recall = 0

    for i in range(targets.shape[0]):
        class_a = targets[i]

        if i % 500 == 0 and i != 0:
            print("iteration: ", i, " mAP: ", sum_av_precision / (i+1), " top 1: ", sum_top1 / (i+1), " top 500: ", sum_top500 / (i+1), " bottom 999: ", sum_bottom1000 / (i+1))
            print("recall: ", sum_recall/(i+1))
            print()

        sorted_hammings = get_hamming(i, binaries)

        ap, recall = average_precision(sorted_hammings, class_a)
        sum_av_precision += ap
        sum_recall += recall

        index_top1 = sorted_hammings[0][0]

        if COARSE_LABELS:
            n = 250
        else:
            n = 50

        index_500 = sorted_hammings[n][0]

        if COARSE_LABELS:
            n = 498
        else:
            n = 98

        index_1000 = sorted_hammings[n][0]

        if class_a == targets[index_top1]:
            sum_top1 += 1

        if class_a == targets[index_500]:
            sum_top500 += 1

        if class_a == targets[index_1000]:
            sum_bottom1000 += 1

    print()
    mean_top_1 = sum_top1 / targets.shape[0]
    print("mean top 1: ", mean_top_1)

    mean_top_500 = sum_top500 / targets.shape[0]
    print("precision 50: ", mean_top_500)

    mean_1000 = sum_bottom1000 / targets.shape[0]
    print("precision 100: ", mean_1000)

    print("recall : ", sum_recall / targets.shape[0])

    map = sum_av_precision / targets.shape[0]
    print("mAP: ", map)


def save_image(original_image, idx):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images/top32_{idx}.png", sample)


binaries = get_binaries(X_test)
get_mAP_and_Top1(binaries)























