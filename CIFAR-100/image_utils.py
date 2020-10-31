
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


horiz_flip = transforms.RandomHorizontalFlip(0.5)
jitter = transforms.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.45, hue=0.45)


def gaussian_blur(X):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    radius = 2
    if random.uniform(0, 1) > 0.5:
        radius = 1

    transformation = transforms.Compose([GaussianSmoothing([0, radius])])
    trans = transforms.Compose([horiz_flip, jitter, transformation, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


# def guassian_noise(X):
#     X_copy = copy.deepcopy(X)
#     X_copy = Variable(torch.FloatTensor(X_copy))
#
#     for i in range(X_copy.shape[0]):
#         transformation = AddGaussianNoise(0., 0.01)
#         trans = transforms.Compose([transforms.ToTensor(), transformation])
#         a = F.to_pil_image(X_copy[i])
#         X_copy[i] = trans(a)
#
#     return X_copy


def no_jitter_rotate(X, degrees):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    # transformation = transforms.RandomRotation(degrees=[-degrees, degrees], fill=(0,))
    transformation = transforms.RandomRotation(degrees=[-degrees, degrees])
    trans = transforms.Compose([horiz_flip, transformation, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def rotate(X, degrees):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    # transformation = transforms.RandomRotation(degrees=[-degrees, degrees], fill=(0,))
    transformation = transforms.RandomRotation(degrees=[-degrees, degrees])
    trans = transforms.Compose([horiz_flip, jitter, transformation, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
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
    X_put = torch.zeros([batch_size, X.shape[1], size, size_y])

    transformation = transforms.Resize(size=(size, size_y))
    trans = transforms.Compose([transformation, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_put[i] = trans(a)

    return X_put


def just_scale(X, size, pad):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    # if random.uniform(0, 1) > 0.5:
    #     size = 20
    #     pad = 4
    transformation = transforms.Resize(size=size, interpolation=2)
    trans = transforms.Compose([transformation, transforms.Pad(pad), transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def scale(X, size, pad, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    # if random.uniform(0, 1) > 0.5:
    #     size = 20
    #     pad = 4

    transformation = transforms.Resize(size=size, interpolation=2)
    trans = transforms.Compose([horiz_flip, jitter, transformation, transforms.Pad(pad), transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def sobel_total(X, batch_size, one_dim=False):
    G_x = sobel_filter_x(X, batch_size, one_dim)
    G_y = sobel_filter_y(X, batch_size, one_dim)

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

    return G.expand(G.shape[0], X.shape[1], G.shape[2], G.shape[3])


def sobel_filter_y(X, batch_size, one_dim=False):
    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])

    b = b.view((1, 1, 3, 3)).expand(1, X.shape[1], 3, 3)

    conv2 = nn.Conv2d(in_channels=X.shape[1], out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(b)

    G_y = conv2(Variable(X)).data.view(batch_size, 1, X.shape[2],  X.shape[3])

    return G_y.expand(G_y.shape[0], X.shape[1], G_y.shape[2], G_y.shape[3])


def sobel_filter_x(X, batch_size, one_dim=False):
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])

    a = a.view((1, 1, 3, 3)).expand(1, X.shape[1], 3, 3)

    conv1 = nn.Conv2d(in_channels=X.shape[1], out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(a)

    G_x = conv1(Variable(X)).data.view(batch_size, 1, X.shape[2], X.shape[3])

    return G_x.expand(G_x.shape[0], X.shape[1], G_x.shape[2], G_x.shape[3])


def vertical_flip(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    transformation = transforms.RandomVerticalFlip(1)
    trans = transforms.Compose([jitter, transformation, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def horizontal_flip(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    transformation = transforms.RandomHorizontalFlip(1)
    trans = transforms.Compose([transformation, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

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

    transformation = RandomErasing()
    trans = transforms.Compose([horiz_flip, jitter, transforms.ToTensor(), transformation])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def center_crop(X, size, batch_size):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    X_res = torch.zeros([batch_size, 1, size, size])
    transformation = transforms.CenterCrop(size=size)

    for i in range(X_copy.shape[0]):

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


def no_jitter_random_corpse(X, size, batch_size, size_y, up_size_x, up_size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    random_crop = transforms.RandomCrop(size=(size, size_y))
    resize = transforms.Resize(size=(up_size_x, up_size_y))

    trans = transforms.Compose([horiz_flip, random_crop, resize, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def randcom_crop_upscale(X, size, batch_size, size_y, up_size_x, up_size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    random_crop = transforms.RandomCrop(size=(size, size_y))
    resize = transforms.Resize(size=(up_size_x, up_size_y))

    trans = transforms.Compose([horiz_flip, jitter, random_crop, resize, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def rc_upscale_rotate(X, size, batch_size, size_y, up_size_x, up_size_y, degrees):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    random_crop = transforms.RandomCrop(size=(size, size_y))
    resize = transforms.Resize(size=(up_size_x, up_size_y))
    rotate = transforms.RandomRotation(degrees=[-degrees, degrees])

    trans = transforms.Compose([horiz_flip, jitter, random_crop, resize, rotate, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def randcom_crop_upscale_gauss_blur(X, size, batch_size, size_y, up_size_x, up_size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    radius = 2
    if random.uniform(0, 1) > 0.5:
        radius = 1

    random_crop = transforms.RandomCrop(size=(size, size_y))
    resize = transforms.Resize(size=(up_size_x, up_size_y))
    gaussian = transforms.Compose([GaussianSmoothing([0, radius])])

    trans = transforms.Compose([horiz_flip, jitter, random_crop, resize, gaussian, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def just_random_crop(X, size, batch_size, size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    X_res = torch.zeros([batch_size, X.shape[1], size, size_y])

    transformation = transforms.RandomCrop(size=(size, size_y))
    trans = transforms.Compose([transformation, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_res[i] = trans(a)

    return X_res


def random_crop(X, size, batch_size, size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    X_res = torch.zeros([batch_size, X.shape[1], size, size_y])

    transformation = transforms.RandomCrop(size=(size, size_y))
    trans = transforms.Compose([horiz_flip, jitter, transformation, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_res[i] = trans(a)

    return X_res


def to_tensor(X):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    trans = transforms.Compose([transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def balanced_subsample(y, size=None):
    subsample = []

    for i in range(100):
        samples = np.where(y == i)
        indexes = np.random.choice(samples[0].size, size=2, replace=False)

        subsample.append(samples[0][indexes][0])
        subsample.append(samples[0][indexes][1])

    return subsample


def preproccess_cifar(x):

    with torch.no_grad():
        x = Variable(torch.FloatTensor(x))

    x = x.transpose(1, 3)
    x = x.transpose(2, 3)

    #x = to_tensor(x)

    #x = rgb2gray(x)
    #x = x.unsqueeze(0)
    #x = x.transpose(0, 1)

    x /= 255

    return x


def color_jitter(X):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    trans = transforms.Compose([horiz_flip, jitter, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

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


def count_common_elements(p, embedings_size):
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

    print("Mean common elements: ", (sum_commons / embedings_size) / counter)


def fix_sobel(sobeled, quarter, image, size_x, size_y):

    AA = sobeled.reshape(sobeled.size(0), sobeled.size(1) * sobeled.size(2) * sobeled.size(3))
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(quarter, image.shape[1], size_x, size_y)

    return AA


def transformation(id, image, size_x, size_y):
    quarter = image.shape[0]

    if id == 0:
        return color_jitter(image)

    elif id == 1:
        return scale(image, (image.shape[2] - 8, image.shape[3] - 8), 4, quarter)

    elif id == 2:
        return rotate(image, 46)

    elif id == 3:
        return torch.abs(1 - color_jitter(image))

    elif id == 4:
        sobeled = sobel_total(color_jitter(image), quarter)
        AA = fix_sobel(sobeled, quarter, image, size_x, size_y)

        return AA

    elif id == 5:
        sobeled = sobel_filter_x(color_jitter(image), quarter)
        AA = fix_sobel(sobeled, quarter, image, size_x, size_y)

        return AA

    elif id == 6:
        sobeled = sobel_filter_y(color_jitter(image), quarter)
        AA = fix_sobel(sobeled, quarter, image, size_x, size_y)

        return AA

    elif id == 7:
        return gaussian_blur(image)

    elif id == 8:
        blured = randcom_crop_upscale_gauss_blur(image, 22, quarter, 22, 32, 32)

        return blured

    elif id == 9:
        scaled_up = randcom_crop_upscale(image, 22, quarter, 22, 32, 32)
        sobeled = sobel_total(scaled_up, quarter)
        AA = fix_sobel(sobeled, quarter, image, size_x, size_y)

        return AA

    elif id == 10:
        scaled_up = randcom_crop_upscale(image, 22, quarter, 22, 32, 32)
        rev = torch.abs(1 - scaled_up)
        return rev

    elif id == 11:
        rot = no_jitter_rotate(image, -46)
        return rot

    elif id == 12:
        scaled_up = randcom_crop_upscale(image, 20, quarter, 20, 32, 32)
        return scaled_up

    elif id == 13:
        scaled_up = randcom_crop_upscale(image, 22, quarter, 22, 32, 32)

        return scaled_up

    elif id == 14:
        scaled_up = randcom_crop_upscale(image, 26, quarter, 26, 32, 32)
        return scaled_up

    elif id == 15:
        scaled_up = no_jitter_random_corpse(image, 22, quarter, 22, 32, 32)

        return scaled_up

    elif id == 16:
        scaled_up = randcom_crop_upscale(image, 18, quarter, 18, 32, 32)
        return scaled_up

    print("Error in transformation of the image.")
    input()
    return image
