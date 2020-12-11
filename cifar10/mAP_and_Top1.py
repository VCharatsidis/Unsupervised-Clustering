
from stl_utils import *
import cifar10_utils
import sys

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(sci_mode=False)

EMBEDDINGS = 1024
name = "binary_contrast_4_"+str(EMBEDDINGS)+"_2.model"
encoder = torch.load(name)
print("encoder: ", name)
encoder.eval()

X_train_raw, y_train_raw, X_test_raw, y_test_raw = cifar10_utils.load_cifar10(cifar10_utils.CIFAR10_FOLDER)
_, _, X_test, targets = cifar10_utils.preprocess_cifar10_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw)

X_test = torch.from_numpy(X_test)
X_test /= 255

PA = 1000


def get_binaries(X_test):
    size = 500
    runs = len(X_test) // size

    binaries = torch.zeros(size, EMBEDDINGS)
    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)
        test_ids = np.array(test_ids)

        image = X_test[test_ids]

        with torch.no_grad():
            _, _, binary = encoder(image.cuda())
            binary = torch.round(binary)
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
    :return: array with hammings and indexes sorted.
    '''
    with torch.no_grad():
        hummings = torch.logical_xor(binaries[idx], binaries[ids]).sum(dim=1)

    sorted, indices = torch.sort(hummings)

    s = [(indices[i].item(), sorted[i].item()) for i in range(PA+1)]
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

    ap = sum_precisions / PA
    return ap


def get_mAP_and_Top1(binaries):
    sum_top1 = 0
    sum_ap = 0

    for i in range(targets.shape[0]):
        class_a = targets[i]

        if i % 100 == 0 and i != 0:
            print("iteration: ", i, " mAP: ", sum_ap / i, " top 1: ", sum_top1 / i)

        sorted_hammings = get_hamming(i, binaries)

        ap = average_precision(sorted_hammings, class_a)

        index_top1 = sorted_hammings[0][0]

        if class_a == targets[index_top1]:
            sum_top1 += 1

        sum_ap += ap


    mean_top_1 = sum_top1 / targets.shape[0]
    print("mean top 1: ", mean_top_1)

    map = sum_ap / targets.shape[0]
    print("mAP: ", map)


binaries = get_binaries(X_test)
get_mAP_and_Top1(binaries)























