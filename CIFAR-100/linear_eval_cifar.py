from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from stl_utils import *

from torchvision.utils import make_grid
import matplotlib
from linear_net_cifar import LinearNetCifar
import sys
import pickle
# Default constants

LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 500000

BATCH_SIZE_DEFAULT = 200
USE_EMBEDDING = False

INPUT_NET = 2304
if USE_EMBEDDING:
    INPUT_NET = 4096

PRINT = False and USE_EMBEDDING
ROUND = True and USE_EMBEDDING and not PRINT
PRODUCT = False and USE_EMBEDDING
AGREEMENT = False and USE_EMBEDDING


SIZE = 32
NETS = 1
EVAL_FREQ_DEFAULT = 100
PATIENCE = 100


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

FLAGS = None

encoder_name = "cifar100_models\\alex_disentangle"

encoder = torch.load(encoder_name+".model")
encoder.eval()
#print(list(encoder.brain[0].weight))


if AGREEMENT:
    fileName = "..\\data\\stl10_binary\\train_X.bin"
    X_train = read_all_images(fileName)
    anchors_ids = range(0, 1000)
    anchor_images = torch.from_numpy(X_train[anchors_ids, :])


    def make_anchors(anchor_images):
        x_train = rgb2gray(anchor_images)
        x_tensor = to_tensor(x_train)
        x_tensor = x_tensor.unsqueeze(0)
        anchor_images = x_tensor.transpose(0, 1)
        anchor_images = anchor_images.transpose(2, 3)

        anchor_images = anchor_images / 255.0

        pad = (96 - SIZE) // 2
        anchor_images = scale(anchor_images, SIZE, pad, BATCH_SIZE_DEFAULT)
        anchor_images = anchor_images[:, :, pad:96 - pad, pad:96 - pad]

        with torch.no_grad():
            _, _, anchor_predictions = encoder(anchor_images.to('cuda'))

        return anchor_predictions


    anchor_predictions = make_anchors(anchor_images)
    print(anchor_predictions.shape)

    for i in range(2):
        anchors_ids = range(i * 1000 + 1000, (i + 1) * 1000 + 1000)
        anchor_images = torch.from_numpy(X_train[anchors_ids, :])
        z = make_anchors(anchor_images)
        anchor_predictions = torch.cat([anchor_predictions, z], dim=0)

    print(anchor_predictions.shape)


    INPUT_NET = anchor_predictions.shape[0]


ELEMENTS_EXCEPT_DIAG = BATCH_SIZE_DEFAULT * (BATCH_SIZE_DEFAULT - 1)
square = torch.ones(BATCH_SIZE_DEFAULT, BATCH_SIZE_DEFAULT)
zero_diag = square.fill_diagonal_(0)

############ UNSUPERVISED INFO #############################

batch_size = 100
lr = 1e-4

augments_compared = 2
heads = 1

#encoder_name = "..\\fix_binaries\\stl_binary"


loss = nn.CrossEntropyLoss()


DESCRIPTION = ["Supervised NET with 75% augments per batch."]

DESCRIPTION = ["LOSS: product loss multiplied by mean ", " Image size: " + str(SIZE)\
              +",  BATCH SIZE: " + str(batch_size)\
              +",  lr: " + str(lr)
              ,",  embedding dim: " + str(INPUT_NET)\
              ,",  augments compared: " + str(augments_compared)\
              +",  heads: " + str(heads)\
              +",  " + encoder_name]


def forward_block(X, ids, classifier, optimizer, train, targets):
    images = X[ids, :]
    #images = to_tensor(images)

    if PRODUCT:
        rotated = rotate(images, 46)

        color_jit_image = color_jitter(images)
        sobeled = sobel_total(color_jit_image, BATCH_SIZE_DEFAULT)

        AA = sobeled.reshape(sobeled.size(0), sobeled.size(1) * sobeled.size(2) * sobeled.size(3))
        AA -= AA.min(1, keepdim=True)[0]
        AA /= AA.max(1, keepdim=True)[0]
        sobeled = AA.view(BATCH_SIZE_DEFAULT, images.shape[1], SIZE, SIZE)

    with torch.no_grad():
        if PRODUCT:
            rot_encodings, _, p_rot = encoder(rotated.to('cuda'))
            sobeled_encodings, _, p_sobeled = encoder(sobeled.to('cuda'))

        encodings, _, p = encoder(images.to('cuda'))

    if PRODUCT:
        p = p * p_sobeled * p_rot

    if AGREEMENT:
        p = agreement(p)

    if ROUND:
        p = torch.round(p)

    if PRINT:
        print("=========== p =================")
        print("batch mean ones: ",
              (np.where(p.data.cpu().numpy() > 0.5))[0].shape[0] / (p[0].shape[0] * BATCH_SIZE_DEFAULT))

        sum = p.data.cpu().numpy().sum(axis=0)
        #print(sum)

        mean = p.data.cpu().numpy().mean(axis=0)
        #print(mean)

        print("max value: ", np.amax(mean))
        print("max value index: ", np.argmax(mean))
        print("mean value: ", np.mean(mean))
        count_common_elements(p)


        print(p[1].data.cpu().numpy())
        print(np.where(p[1].data.cpu().numpy() > 0.5))
        print(np.where(p[1].data.cpu().numpy() > 0.5)[0].shape)
        #
        # print(p[2].data.cpu().numpy())
        # print(np.where(p[2].data.cpu().numpy() > 0.5))
        # print(np.where(p[2].data.cpu().numpy() > 0.5)[0].shape)


        # common_ones = p[1].data.cpu().numpy() * p[2].data.cpu().numpy()
        # print("comon ones: ", np.where(common_ones > 0.5)[0].shape[0])


    # product = p * p_sobeled

    # print("=========== product =================")
    # print(product.data.cpu().numpy().sum(axis=0))
    #
    # print(product[1].data.cpu().numpy())
    # print(np.where(product[1].data.cpu().numpy() > 0.5))
    # print(np.where(product[1].data.cpu().numpy() > 0.5)[0].shape)
    #
    # print(product[2].data.cpu().numpy())
    # print(np.where(product[2].data.cpu().numpy() > 0.5))
    # print(np.where(product[2].data.cpu().numpy() > 0.5)[0].shape)

    #concat_encodings = torch.cat([encodings, sobeled_encodings], dim=1)


    #concat = torch.cat([p, p_sobeled], dim=1)
    if USE_EMBEDDING:
        preds = classifier(p.detach())
    else:
        preds = classifier(encodings.detach())

    targets_tensor = Variable(torch.LongTensor(targets[ids])).cuda()

    cross_entropy_loss = loss(preds, targets_tensor)

    if train:

        # for p in encoder.parameters():
        #     p.requires_grad = False

        optimizer.zero_grad()
        cross_entropy_loss.backward()
        optimizer.step()

    return preds, cross_entropy_loss


def count_common_elements(p):
    sum_commons = 0
    counter = 0
    for i in range(p.shape[0]):
        for j in range(p.shape[0]):
            if i == j:
                continue

            product = p[i].data.cpu().numpy() * p[j].data.cpu().numpy()
            commons = np.where(product > 0.5)[0].shape[0]
            #print(commons)

            sum_commons += commons
            counter += 1

    print("Mean common elements: ", (sum_commons / INPUT_NET) / counter)



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


def agreement(preds_1):
    transposed = anchor_predictions.transpose(0, 1)
    nondiag = torch.mm(preds_1, transposed)

    nondiag = nondiag / anchor_predictions.sum(dim=1)

    #log_nondiag = - torch.log(nondiag)

    #cleaned = nondiag * zero_diag.cuda()

    #negative = cleaned.sum(dim=0).sum(dim=0) / ELEMENTS_EXCEPT_DIAG

    return nondiag


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


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def preproccess_cifar(x, channels=3):
    x = to_tensor(x)

    x = x.transpose(1, 3)
    x = x.transpose(2, 3)

    # if channels == 1:
    #     x = rgb2gray(x)
    #     x = x.unsqueeze(0)
    #     x = x.transpose(0, 1)

    #x /= 255

    return x

def train():
    global EVAL_FREQ_DEFAULT
    with open('data\\train', 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')

    meta = unpickle('data\\meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]

    train = unpickle('data\\train')

    filenames = [t.decode('utf8') for t in train[b'filenames']]
    y_train = train[b'fine_labels']
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
    y_train = np.array(y_train)
    print("y train shape", y_train.shape)

    script_directory = os.path.split(os.path.abspath(__file__))[0]

    filepath = 'encoders\\encoder_' + str(0) + '.model'
    path_to_model = os.path.join(script_directory, filepath)

    linearClassifier = LinearNetCifar(INPUT_NET).cuda()
    optimizer = torch.optim.Adam(linearClassifier.parameters(), lr=LEARNING_RATE_DEFAULT)

    print("X train shape: ", X_train.shape, " train y shape: ", y_train.shape, " X test shape: ", X_test.shape, " X test y:", targets.shape)
    best_accuracy = 0
    iter_acc = 0
    patience = 50

    for iteration in range(MAX_STEPS_DEFAULT):
        linearClassifier.train()
        ids = np.random.choice(len(X_train), size=BATCH_SIZE_DEFAULT, replace=False)
        train = True
        preds, mim = forward_block(X_train, ids, linearClassifier, optimizer, train, y_train)

        if iteration > 4000:
            EVAL_FREQ_DEFAULT = 50

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
                break


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
