
from image_utils import *
import cifar10_utils
import matplotlib
from torchvision.utils import make_grid
from sklearn.datasets import fetch_openml
import sys

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)

torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(sci_mode=False)

EMBEDDINGS = 64
#name = "..\\CIFAR-100\\cifar100_models\\simple_penalty_4_400.model"
#name = "binary_contrast_4_"+str(EMBEDDINGS)+"_b2_1.model"
name = "binary_contrast_4_4096_2.model"
encoder = torch.load(name)
print("encoder: ", name)
encoder.eval()

mnist = fetch_openml('mnist_784', version=1, cache=True)
targets = mnist.target[60000:]
targets = targets.astype(np.int64)
#targets = torch.from_numpy(targets)

X_train = mnist.data[:60000]
X_test = mnist.data[60000:]

y_train = mnist.target[:60000]
y_train = y_train.astype(np.int64)
#y_train = torch.from_numpy(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
X_train = np.stack((X_train,)*3, axis=1)
X_train = np.squeeze(X_train, axis=2)
X_train= Variable(torch.FloatTensor(X_train))
X_train /= 255
X_train = just_scale_up(X_train, 32)

X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))
X_test = np.stack((X_test,) * 3, axis=1)
X_test = np.squeeze(X_test, axis=2)
X_test = Variable(torch.FloatTensor(X_test))
X_test /= 255
X_test = just_scale_up(X_test, 32)

PA = 1000


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
    :param idx: The index of the query image
    :param binaries: the binaries.
    :return: array with hammings and indexes sorted for the given image vs the hole test-set.
    '''

    with torch.no_grad():
        hammings = torch.logical_xor(binaries[idx], binaries[ids]).sum(dim=1)

    sorted, indices = torch.sort(hammings)

    if False:
        print("image")
        save_image(X_test[idx].unsqueeze(dim=0), str(targets[idx]) + "_mnist_q")
        save_image(X_test[indices[1:17]], str(targets[idx]) + "_mnist")

    s = [(indices[i].item(), sorted[i].item()) for i in range(PA)]  # make it a list and keep only the first top PA
    s = s[1:]  # Delete the first element which is the query image itself.

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
        index_500 = sorted_hammings[500][0]
        index_1000 = sorted_hammings[998][0]

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
    print("precision 500: ", mean_top_500)

    mean_1000 = sum_bottom1000 / targets.shape[0]
    print("precision 1000: ", mean_1000)

    print("recall : ", sum_recall / targets.shape[0])

    map = sum_av_precision / targets.shape[0]
    print("mAP: ", map)




def save_image(original_image, idx):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images/{EMBEDDINGS}bits/top16_{idx}.png", sample)


binaries = get_binaries(X_test)
get_mAP_and_Top1(binaries)























