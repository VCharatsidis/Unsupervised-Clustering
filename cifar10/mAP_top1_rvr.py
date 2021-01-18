
from stl_utils import *
import cifar10_utils
import matplotlib
from torchvision.utils import make_grid
import sys

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)

torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(sci_mode=False)

EMBEDDINGS = 64
#name = "..\\CIFAR-100\\cifar100_models\\binary_contrast_4_plus0_64_2.model"
#name = "binary_contrast_4_"+str(EMBEDDINGS)+"_b2_1.model"
name = "binary_contrast_4_64_b2_1.model"
encoder = torch.load(name)
print("encoder: ", name)
encoder.eval()

X_train_raw, y_train_raw, X_test_raw, y_test_raw = cifar10_utils.load_cifar10(cifar10_utils.CIFAR10_FOLDER)
X_Train, y_train, X_test, targets = cifar10_utils.preprocess_cifar10_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw)

X_test = torch.from_numpy(X_test)
X_test /= 255

X_Train = torch.from_numpy(X_Train)
X_Train /= 255

all = torch.cat([X_test, X_Train], dim=0)
all_targets = np.concatenate((targets, y_train), axis=0)

ids = np.random.choice(60000, size=10000, replace=False)
print(ids)

g_ids = [x for x in range(60000) if x not in ids]
#g_ids = np.array(g_ids)

print("ids shape ", ids.shape)
print("galery ids shape ", g_ids)

print("all shape: ", all.shape)
print("all_targets shape: ", all_targets.shape)

query_images = all[ids]
galery_images = all[g_ids]

y_train = all_targets[ids]
targets = all_targets[g_ids]



PA = 1000


def get_binaries(images):
    size = 500
    runs = len(images) // size

    for j in range(runs):
        test_ids = range(j * size, (j + 1) * size)
        test_ids = np.array(test_ids)

        image = images[test_ids]

        with torch.no_grad():
            _, _, binary = encoder(image.cuda())

            binary = torch.round(binary)  # round the prediction to make it real binary.

            if j == 0:
                binaries = binary
            else:
                binaries = torch.cat([binaries, binary], dim=0)

    print("Binaries shape: ", binaries.shape)
    return binaries


def get_hamming(idx, bin_query, bin_galery):
    '''
    Gets the index of 1 image and XOR it's binary with all other binaries of the test-set
    :param idx: The index of the query image
    :param binaries: the binaries.
    :return: array with hammings and indexes sorted for the given image vs the hole test-set.
    '''

    with torch.no_grad():
        hammings = torch.logical_xor(bin_query[idx], bin_galery).sum(dim=1)

    sorted, indices = torch.sort(hammings)

    if False:
        save_image(X_test[idx].unsqueeze(dim=0), str(idx) + "q")
        save_image(X_test[indices[1:17]], idx)

    s = [(indices[i].item(), sorted[i].item()) for i in range(PA)]  # make it a list and keep only the first top PA

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


def get_mAP_and_Top1(bin_querry, bin_galery):
    sum_top1 = 0
    sum_top500 = 0
    sum_bottom1000 = 0
    sum_av_precision = 0

    sum_recall = 0

    for i in range(y_train.shape[0]):
        class_a = y_train[i]

        if i % 500 == 0 and i != 0:
            print("iteration: ", i, " mAP: ", sum_av_precision / (i+1), " top 1: ", sum_top1 / (i+1), " top 500: ", sum_top500 / (i+1), " bottom 999: ", sum_bottom1000 / (i+1))
            print("recall: ", sum_recall/(i+1))
            print()

        sorted_hammings = get_hamming(i, bin_querry,  bin_galery)

        ap, recall = average_precision(sorted_hammings, class_a)
        sum_av_precision += ap
        sum_recall += recall

        index_top1 = sorted_hammings[0][0]
        index_500 = sorted_hammings[500][0]
        index_1000 = sorted_hammings[999][0]

        if class_a == targets[index_top1]:
            sum_top1 += 1

        if class_a == targets[index_500]:
            sum_top500 += 1

        if class_a == targets[index_1000]:
            sum_bottom1000 += 1

    print()
    mean_top_1 = sum_top1 / y_train.shape[0]
    print("mean top 1: ", mean_top_1)

    mean_top_500 = sum_top500 / y_train.shape[0]
    print("precision 500: ", mean_top_500)

    mean_1000 = sum_bottom1000 / y_train.shape[0]
    print("precision 1000: ", mean_1000)

    print("recall : ", sum_recall / y_train.shape[0])

    map = sum_av_precision / y_train.shape[0]
    print("mAP: ", map)




def save_image(original_image, idx):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images/{EMBEDDINGS}bits/top16_{idx}.png", sample)


binaries_querry = get_binaries(query_images)
binaries_galery = get_binaries(galery_images)

get_mAP_and_Top1(binaries_querry, binaries_galery)























