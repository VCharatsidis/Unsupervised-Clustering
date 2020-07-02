
from torchvision import transforms
import torchvision.transforms.functional as F
import copy
import numpy as np
from torch.autograd import Variable
import torch
from Mutual_Information.RandomErase import RandomErasing
from GaussianBlur import GaussianSmoothing
import torch.nn as nn
import matplotlib.pyplot as plt
import random
BATCH_SIZE_DEFAULT = 100


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def gaussian_blur(X):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        transformation = transforms.Compose([GaussianSmoothing([0, 2])])
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def guassian_noise(X):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        transformation = AddGaussianNoise(0., 0.01)
        trans = transforms.Compose([transforms.ToTensor(), transformation])
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def rotate(X, degrees):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        transformation = transforms.RandomRotation(degrees=[-degrees, degrees],  fill=(0,))
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def vertical_blacken(X):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    size = X_copy.shape[3]
    crop = size//2 + size//8

    if random.uniform(0, 1) > 0.5:
        X_copy[:, :, :, :(size-crop)] = 0
    else:
        X_copy[:, :, :, crop:] = 0

    return X_copy

def horizontal_blacken(X):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    size = X_copy.shape[2]
    crop = size//2 + size//5

    if random.uniform(0, 1) > 0.5:
        X_copy[:, :, :(size-crop), :] = 0
    else:
        X_copy[:, :,  crop:, :] = 0

    return X_copy


def binary(images, threshold=120):
    bin = images > threshold/255
    bin = bin.float()

    return bin


def rgb2gray(rgb):

    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def random_derangement(n):
    while True:
        v = list(range(n))
        for j in list(range(n - 1, -1, -1)):
            p = random.randint(0, j)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return tuple(v)


def upscale(X, size):
    X_copy = Variable(torch.FloatTensor(BATCH_SIZE_DEFAULT, 3, 224, 224))

    for i in range(X_copy.shape[0]):
        transformation = transforms.Resize(224)
        trans = transforms.Compose([transformation, transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225])])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def scale_up(X, size, size_y, batch_size):
    X_copy = copy.deepcopy(X)
    X_put = torch.zeros([batch_size, 1, size, size_y])

    for i in range(X_copy.shape[0]):
        transformation = transforms.Resize(size=(size, size_y))
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_put[i] = trans_image

    return X_put


def scale(X, size, pad, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)

    X_copy = Variable(torch.FloatTensor(X_copy))

    # if random.uniform(0, 1) > 0.5:
    #     size = 20
    #     pad = 4

    for i in range(X_copy.shape[0]):
        transformation = transforms.Resize(size=size, interpolation=2)
        trans = transforms.Compose([transformation, transforms.Pad(pad), transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def sobel_total(X, batch_size):
    G_x = sobel_filter_x(X, batch_size)
    G_y = sobel_filter_y(X, batch_size)

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

    return G


def sobel_filter_y(X, batch_size):
    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])

    b = b.view((1, 1, 3, 3))

    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(b)

    G_y = conv2(Variable(X)).data.view(batch_size, 1, X.shape[2],  X.shape[3])

    return G_y


def sobel_filter_x(X, batch_size):
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])

    a = a.view((1, 1, 3, 3))

    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(a)

    G_x = conv1(Variable(X)).data.view(batch_size, 1, X.shape[2], X.shape[3])

    return G_x


def vertical_flip(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        transformation = transforms.RandomVerticalFlip(1)
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def horizontal_flip(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        transformation = transforms.RandomHorizontalFlip(1)
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def to_grayscale(X, channels, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        transformation = transforms.Grayscale(num_output_channels=channels)
        trans = transforms.Compose([transformation, transforms.ToTensor()])

        a = F.to_pil_image(X_copy[i])

        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def random_erease(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        transformation = RandomErasing()
        trans = transforms.Compose([transforms.ToTensor(), transformation])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def center_crop(X, size, batch_size):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    X_res = torch.zeros([batch_size, 1, size, size])

    for i in range(X_copy.shape[0]):
        transformation = transforms.CenterCrop(size=size)
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_res[i] = trans_image

    return X_res


def add_noise(X, batch_size=BATCH_SIZE_DEFAULT, max_noise_percentage=0.3):
    X_copy = copy.deepcopy(X)
    threshold = random.uniform(0.1, max_noise_percentage)

    for i in range(X_copy.shape[0]):
        nums = np.random.uniform(low=0, high=1, size=(X_copy[i].shape[0],))
        X_copy[i] = np.where(nums > threshold, X_copy[i], 0)

    return to_tensor(X_copy)


def to_tensor(X):
    with torch.no_grad():
        X = Variable(torch.FloatTensor(X))

    return X


def random_crop(X, size, batch_size, size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    X_res = torch.zeros([batch_size, 1, size, size_y])

    for i in range(X_copy.shape[0]):
        transformation = transforms.RandomCrop(size=(size, size_y))
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_res[i] = trans_image

    return X_res


def color_jitter(X):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        transformation = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.35)
        trans = transforms.Compose([transformation, transforms.ToTensor()])
        a = F.to_pil_image(X_copy[i])
        trans_image = trans(a)
        X_copy[i] = trans_image

    return X_copy


def show_gray_numpy(image_1):
    z = image_1.squeeze(1)
    pixels = z[0]
    plt.imshow(pixels, cmap='gray')
    plt.show()


def show_gray(image_1):
    z = image_1
    print(z.shape)
    if len(list(z.size())) == 4:
        z = image_1.squeeze(1)

    pixels = z[0]
    plt.imshow(pixels, cmap='gray')
    plt.show()


def show_image(image_1):
    image_1 = torch.transpose(image_1, 1, 3)
    show_mnist(image_1[0], image_1[0].shape[1], image_1[0].shape[2])
    image_1 = torch.transpose(image_1, 1, 3)

    return image_1


def show_mnist(first_image, w, h):
    #pixels = first_image.reshape((w, h))
    pixels = first_image
    plt.imshow(pixels)
    plt.show()



