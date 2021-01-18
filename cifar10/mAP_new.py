
from stl_utils import *
import cifar10_utils
import matplotlib
from torchvision.utils import make_grid
from excluded_ids import exclude
import sys

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
np.set_printoptions(threshold=sys.maxsize)

torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(sci_mode=False)

PRINT = False
EMBEDDINGS = 64
#name = "..\\CIFAR-100\\cifar100_models\\binary_contrast_4_plus0_64_2.model"
#name = "binary_contrast_4_"+str(EMBEDDINGS)+"_b2_1.model"
name = "binary_contrast_4_64_b5_lr0.00025_1.model"
encoder = torch.load(name)
print("encoder: ", name)
encoder.eval()

X_train_raw, y_train_raw, X_test_raw, y_test_raw = cifar10_utils.load_cifar10(cifar10_utils.CIFAR10_FOLDER)
_, _, X_test, gallery_targets = cifar10_utils.preprocess_cifar10_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw)

galery_images = torch.from_numpy(X_test)
galery_images /= 255

_, _, query_images, query_targets = exclude()

print("query images shape: ", query_images.shape)
print("galery images shape: ", galery_images.shape)

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

    if PRINT:
        print(idx)
        save_image(query_images[idx].unsqueeze(dim=0), str(idx) + "q")
        save_image(galery_images[indices[0:16]], idx)

    s = [(indices[i].item(), sorted[i].item()) for i in range(PA)]  # make it a list and keep only the first top PA

    return s


def average_precision(hammings, class_a):
    corrects = 0
    sum_precisions = 0
    for h in range(len(hammings)):
        index = hammings[h][0]
        c = gallery_targets[index]

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

    for i in range(query_targets.shape[0]):
        class_a = query_targets[i]

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

        if class_a == gallery_targets[index_top1]:
            sum_top1 += 1

        if class_a == gallery_targets[index_500]:
            sum_top500 += 1

        if class_a == gallery_targets[index_1000]:
            sum_bottom1000 += 1

    print()
    mean_top_1 = sum_top1 / query_targets.shape[0]
    print("mean top 1: ", mean_top_1)

    mean_top_500 = sum_top500 / query_targets.shape[0]
    print("precision 500: ", mean_top_500)

    mean_1000 = sum_bottom1000 / query_targets.shape[0]
    print("precision 1000: ", mean_1000)

    print("recall : ", sum_recall / query_targets.shape[0])

    map = sum_av_precision / query_targets.shape[0]
    print("mAP: ", map)


def save_image(original_image, idx):
    sample = original_image.view(-1, original_image.shape[1], original_image.shape[2], original_image.shape[3])
    sample = make_grid(sample, nrow=8).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images/{EMBEDDINGS}bits/top16_{idx}.png", sample)


binaries_querry = get_binaries(query_images)
binaries_galery = get_binaries(galery_images)

get_mAP_and_Top1(binaries_querry, binaries_galery)























