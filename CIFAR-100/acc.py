import argparse
import os

from stl_utils import *

from torchvision.utils import make_grid
import matplotlib
from linear_net import LinearNet
import sys
import pickle

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

FLAGS = None
FINE_LABELS = False

encoder_name = "cifar100_models\\PPA_4_128_4conv_1linear_1"

encoder = torch.load(encoder_name+".model")
encoder.eval()

CLASSES = 100
BATCH_SIZE_DEFAULT = 500

labels_to_imags = {}
for i in range(CLASSES):
    labels_to_imags[i] = i

def miss_classifications(cluster):
    mfe = most_frequent(cluster)
    missclassifications = 0
    for j in cluster:
        if j != mfe:
            missclassifications += 1

    return missclassifications


def most_frequent(List):
    return max(set(List), key=List.count)


def measure_acc_augments(X_test, encoder, targets):
    virtual_clusters = {}
    for i in range(CLASSES):
        virtual_clusters[i] = []

    size = 500
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

        _, _, p = encoder(images.to('cuda'))

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
        if length == 0:
            continue
        misses = miss_classifications(print_dict[element])
        total_miss += misses

        mfe = most_frequent(print_dict[element])
        clusters.add(mfe)
        # print("cluster: ",
        #       labels_to_imags[element],
        #       ", most frequent: ",
        #       mfe,
        #       ", miss-classifications: ",
        #       misses,
        #       ", miss percentage: ",
        #       misses / length)

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
        #virtual_length = len(virtual_clusters[element])

        virtual_misses = miss_classifications(virtual_clusters[element])
        total_miss_virtual += virtual_misses

        #mfe = most_frequent(virtual_clusters[element])

        # print("cluster: ",
        #       element,
        #       ", most frequent: ",
        #       labels_to_imags[mfe],
        #       ", miss-classifications: ",
        #       virtual_misses,
        #       ", miss percentage: ",
        #       virtual_misses / virtual_length)

    miss_virtual_percentage = total_miss_virtual / (runs * size)
    print("acc virtual percentage: ", 1 - miss_virtual_percentage)

    return total_miss_percentage, len(clusters), miss_virtual_percentage


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
    if not FINE_LABELS:
        train_fine_labels = train[b'coarse_labels']
    train_data = train[b'data']

    test = unpickle('data\\test')

    filenames = [t.decode('utf8') for t in test[b'filenames']]
    targets = test[b'fine_labels']
    if not FINE_LABELS:
        targets = test[b'coarse_labels']
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
    y_train = np.array(train_fine_labels)

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

    print("X_train: ", X_train.shape, " X_test: ", X_test.shape, " targets: ", targets.shape)

    measure_acc_augments(X_test, encoder, targets)
    print()
    print()
    print("train")
    measure_acc_augments(X_train, encoder, y_train)


def main():
    """
    Main function
    """
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')

    FLAGS, unparsed = parser.parse_known_args()

    main()
