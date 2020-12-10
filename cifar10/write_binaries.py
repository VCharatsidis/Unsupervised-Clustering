import argparse
import os
from stl_utils import *
import cifar10_utils
from torchvision.utils import make_grid
import matplotlib
from linear_net_cifar_10 import LinearNetCifar10
import sys

import heapq
import random
import time


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

    with torch.no_grad():
        hummings = torch.logical_xor(binaries[idx], binaries[ids]).sum(dim=1)

    sorted, indices = torch.sort(hummings)

    s = [(indices[i].item(), sorted[i].item()) for i in range(PA+1)]
    s = s[1:]

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


binaries = get_binaries(X_test)

#

# def heapSearch( bigArray, k ):
#     heap = []
#     # Note: below is for illustration. It can be replaced by
#     # heapq.nlargest( bigArray, k )
#     for item in bigArray:
#         # If we have not yet found k items, or the current item is larger than
#         # the smallest item on the heap,
#
#         if len(heap) < k or item[1] > heap[0][1]:
#             # if len(heap)>0:
#             #     print(item, item[1], heap[0][1])
#             # If the heap is full, remove the smallest element on the heap.
#             if len(heap) == k:
#                 heapq.heappop(heap)
#             # add the current element as the new smallest.
#             heapq.heappush(heap, item)
#
#     return [heapq.heappop(heap) for i in range(k)]



sum_top1 = 0
sum_ap = 0

for i in range(targets.shape[0]):
    a = binaries[i]
    class_a = targets[i]

    if i % 100 == 0 and i != 0:
        print("iteration: ", i, " mAP: ", sum_ap / i, " top 1: ", sum_top1 / i)

    # start = time.time()
    #
    # hammings = []
    # for j in range(targets.shape[0]):
    #     if i == j:
    #         continue
    #
    #     b = binaries[j]
    #
    #     h = torch.logical_xor(a, b).sum()
    #     h = h.cpu().numpy()
    #
    #     hammings.append((j, h))
    #
    # print("hamming time: ", time.time()-start)
    # print(hammings)

    #start = time.time()

    sorted_hammings = get_hamming(i, binaries)

    #print("hamming time torch: ", time.time() - start)


    #sorted_hammings = sorted(torch_hammings, key=lambda x: x[1])[:PA]

    ap = average_precision(sorted_hammings, class_a)

    index_top1 = sorted_hammings[0][0]

    if class_a == targets[index_top1]:
        sum_top1 += 1

    sum_ap += ap


mean_top_1 = sum_top1 / targets.shape[0]
print("mean top 1: ", mean_top_1)

map = sum_ap / targets.shape[0]
print("mAP: ", map)






def createArray():
    array = range( 10 * 1000 * 1000 )
    random.shuffle( array )
    return array


def linearSearch( bigArray, k ):
    return sorted(bigArray, reverse=True)[:k]





















