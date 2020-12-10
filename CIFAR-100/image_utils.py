
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
    def __init__(self, mean=0.5, std=0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        prob = 0.08

        rnd = torch.rand(tensor.size())
        noisy = tensor[:]
        noisy[rnd < prob / 2] = 0.
        noisy[rnd > 1 - prob / 2] = 1.

        return noisy

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


noise = AddGaussianNoise()
horiz_flip = transforms.RandomHorizontalFlip(0.5)
jitter = transforms.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.45, hue=0.45)
erase = RandomErasing()
resize = transforms.Resize(size=(32, 32))
rd_rotate = transforms.RandomRotation(degrees=[-46, 46])

random_crop_26 = transforms.RandomCrop(size=(26, 26))
random_crop_22 = transforms.RandomCrop(size=(22, 22))
random_crop_20 = transforms.RandomCrop(size=(20, 20))
random_crop_18 = transforms.RandomCrop(size=(18, 18))


trans_color_jitter = transforms.Compose([horiz_flip, jitter, transforms.ToTensor()])
trans_random_erase = transforms.Compose([horiz_flip, jitter, transforms.ToTensor(), erase])
trans_randcom_crop_upscale_18 = transforms.Compose([horiz_flip, jitter, random_crop_18, resize, transforms.ToTensor()])
trans_randcom_crop_upscale_20 = transforms.Compose([horiz_flip, jitter, random_crop_20, resize, transforms.ToTensor()])
trans_no_jitter_random_corpse_22 = transforms.Compose([horiz_flip, random_crop_22, resize, transforms.ToTensor()])
trans_randcom_crop_upscale_22 = transforms.Compose([horiz_flip, jitter, random_crop_22, resize, transforms.ToTensor()])
trans_randcom_crop_upscale_26 = transforms.Compose([horiz_flip, jitter, random_crop_26, resize, transforms.ToTensor()])
trans_scale_up = transforms.Compose([resize, transforms.ToTensor()])
trans_noise = transforms.Compose([horiz_flip, jitter, transforms.ToTensor(),  noise])

b = torch.Tensor([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]])

a = torch.Tensor([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]])


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
    #transformation = transforms.RandomRotation(degrees=[-degrees, degrees])
    trans = transforms.Compose([horiz_flip, rd_rotate, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans(a)

    return X_copy


def rotate(X, degrees):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    # transformation = transforms.RandomRotation(degrees=[-degrees, degrees], fill=(0,))
    #transformation = transforms.RandomRotation(degrees=[-degrees, degrees])
    trans = transforms.Compose([horiz_flip, jitter, rd_rotate, transforms.ToTensor()])

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


def rand_bin_array(K, N):
    arr = np.zeros([N, N])
    arr[:, :K] = 1

    arr = np.take_along_axis(arr, np.random.randn(*arr.shape).argsort(axis=1), axis=1)

    return arr

# def upscale(X, size):
#     X_copy = Variable(torch.FloatTensor(BATCH_SIZE_DEFAULT, 3, 224, 224))
#
#     for i in range(X_copy.shape[0]):
#         transformation = transforms.Resize(224)
#         trans = transforms.Compose([transformation, transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225])])
#         a = F.to_pil_image(X_copy[i])
#         trans_image = trans(a)
#         X_copy[i] = trans_image
#
#     return X_copy


# def scale_up(X, size, size_y, batch_size):
#     X_copy = copy.deepcopy(X)
#     X_put = torch.zeros([batch_size, X.shape[1], size, size_y])
#
#     for i in range(X_copy.shape[0]):
#         a = F.to_pil_image(X_copy[i])
#         X_put[i] = trans_scale_up(a)
#
#     return X_put


# def just_scale(X, size, pad):
#     X_copy = copy.deepcopy(X)
#     X_copy = Variable(torch.FloatTensor(X_copy))
#
#     # if random.uniform(0, 1) > 0.5:
#     #     size = 20
#     #     pad = 4
#     transformation = transforms.Resize(size=size, interpolation=2)
#     trans = transforms.Compose([transformation, transforms.Pad(pad), transforms.ToTensor()])
#
#     for i in range(X_copy.shape[0]):
#         a = F.to_pil_image(X_copy[i])
#         X_copy[i] = trans(a)
#
#     return X_copy


def scale(X, size, pad, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    # if random.uniform(0, 1) > 0.5:
    #     size = 20
    #     pad = 4

    scale = transforms.Resize(size=size, interpolation=2)
    trans = transforms.Compose([horiz_flip, jitter, scale, transforms.Pad(pad), transforms.ToTensor()])

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
    b_view = b.view((1, 1, 3, 3)).expand(1, X.shape[1], 3, 3)

    conv2 = nn.Conv2d(in_channels=X.shape[1], out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(b_view)

    G_y = conv2(Variable(X)).data.view(batch_size, 1, X.shape[2],  X.shape[3])

    return G_y.expand(G_y.shape[0], X.shape[1], G_y.shape[2], G_y.shape[3])


def sobel_filter_x(X, batch_size, one_dim=False):
    a_view = a.view((1, 1, 3, 3)).expand(1, X.shape[1], 3, 3)

    conv1 = nn.Conv2d(in_channels=X.shape[1], out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(a_view)

    G_x = conv1(Variable(X)).data.view(batch_size, 1, X.shape[2], X.shape[3])

    return G_x.expand(G_x.shape[0], X.shape[1], G_x.shape[2], G_x.shape[3])


# def vertical_flip(X, batch_size=BATCH_SIZE_DEFAULT):
#     X_copy = copy.deepcopy(X)
#     X_copy = Variable(torch.FloatTensor(X_copy))
#
#     transformation = transforms.RandomVerticalFlip(1)
#     trans = transforms.Compose([jitter, transformation, transforms.ToTensor()])
#
#     for i in range(X_copy.shape[0]):
#         a = F.to_pil_image(X_copy[i])
#         X_copy[i] = trans(a)
#
#     return X_copy


# def horizontal_flip(X, batch_size=BATCH_SIZE_DEFAULT):
#     X_copy = copy.deepcopy(X)
#     X_copy = Variable(torch.FloatTensor(X_copy))
#
#     transformation = transforms.RandomHorizontalFlip(1)
#     trans = transforms.Compose([transformation, transforms.ToTensor()])
#
#     for i in range(X_copy.shape[0]):
#         a = F.to_pil_image(X_copy[i])
#         X_copy[i] = trans(a)
#
#     return X_copy


# def to_grayscale(X, channels, batch_size=BATCH_SIZE_DEFAULT):
#     X_copy = copy.deepcopy(X)
#     X_copy = Variable(torch.FloatTensor(X_copy))
#
#     for i in range(X_copy.shape[0]):
#         transformation = transforms.Grayscale(num_output_channels=channels)
#         trans = transforms.Compose([transformation, transforms.ToTensor()])
#
#         a = F.to_pil_image(X_copy[i])
#
#         trans_image = trans(a)
#         X_copy[i] = trans_image
#
#     return X_copy


def random_erease(X, batch_size=BATCH_SIZE_DEFAULT):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans_random_erase(a)

    return X_copy


# def center_crop(X, size, batch_size):
#     X_copy = copy.deepcopy(X)
#     X_copy = Variable(torch.FloatTensor(X_copy))
#
#     X_res = torch.zeros([batch_size, 1, size, size])
#     transformation = transforms.CenterCrop(size=size)
#
#     for i in range(X_copy.shape[0]):
#
#         trans = transforms.Compose([transformation, transforms.ToTensor()])
#         a = F.to_pil_image(X_copy[i])
#         trans_image = trans(a)
#         X_res[i] = trans_image
#
#     return X_res


# def add_noise(X, batch_size=BATCH_SIZE_DEFAULT, max_noise_percentage=0.3):
#     X_copy = copy.deepcopy(X)
#     threshold = random.uniform(0.1, max_noise_percentage)
#
#     for i in range(X_copy.shape[0]):
#         nums = np.random.uniform(low=0, high=1, size=X_copy[i].shape[1])
#         X_copy[i] = np.where(nums > threshold, X_copy[i], 0)
#
#     return X_copy


def no_jitter_random_corpse(X, size, batch_size, size_y, up_size_x, up_size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans_no_jitter_random_corpse_22(a)

    return X_copy


def randcom_crop_upscale_26(X, size, batch_size, size_y, up_size_x, up_size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans_randcom_crop_upscale_26(a)

    return X_copy


def randcom_crop_upscale_22(X, size, batch_size, size_y, up_size_x, up_size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans_randcom_crop_upscale_22(a)

    return X_copy


def randcom_crop_upscale_20(X, size, batch_size, size_y, up_size_x, up_size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans_randcom_crop_upscale_20(a)

    return X_copy


def randcom_crop_upscale_18(X, size, batch_size, size_y, up_size_x, up_size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans_randcom_crop_upscale_18(a)

    return X_copy


# def rc_upscale_rotate(X, size, batch_size, size_y, up_size_x, up_size_y, degrees):
#     X_copy = copy.deepcopy(X)
#     X_copy = Variable(torch.FloatTensor(X_copy))
#
#     random_crop = transforms.RandomCrop(size=(size, size_y))
#     #resize = transforms.Resize(size=(up_size_x, up_size_y))
#
#     trans = transforms.Compose([horiz_flip, jitter, random_crop, resize, rotate, transforms.ToTensor()])
#
#     for i in range(X_copy.shape[0]):
#         a = F.to_pil_image(X_copy[i])
#         X_copy[i] = trans(a)
#
#     return X_copy


def randcom_crop_upscale_gauss_blur(X, size, batch_size, size_y, up_size_x, up_size_y):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    radius = 2
    if random.uniform(0, 1) > 0.5:
        radius = 1

    gaussian = transforms.Compose([GaussianSmoothing([0, radius])])

    randcom_crop_upscale_gauss_blur = transforms.Compose([horiz_flip, jitter, random_crop_22, resize, gaussian, transforms.ToTensor()])

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = randcom_crop_upscale_gauss_blur(a)

    return X_copy


# def just_random_crop(X, size, batch_size, size_y):
#     X_copy = copy.deepcopy(X)
#     X_copy = Variable(torch.FloatTensor(X_copy))
#
#     X_res = torch.zeros([batch_size, X.shape[1], size, size_y])
#
#     transformation = transforms.RandomCrop(size=(size, size_y))
#     trans = transforms.Compose([transformation, transforms.ToTensor()])
#
#     for i in range(X_copy.shape[0]):
#         a = F.to_pil_image(X_copy[i])
#         X_res[i] = trans(a)
#
#     return X_res


# def random_crop(X, size, batch_size, size_y):
#     X_copy = copy.deepcopy(X)
#     X_copy = Variable(torch.FloatTensor(X_copy))
#
#     X_res = torch.zeros([batch_size, X.shape[1], size, size_y])
#
#     transformation = transforms.RandomCrop(size=(size, size_y))
#     trans = transforms.Compose([horiz_flip, jitter, transformation, transforms.ToTensor()])
#
#     for i in range(X_copy.shape[0]):
#         a = F.to_pil_image(X_copy[i])
#         X_res[i] = trans(a)
#
#     return X_res


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


def preproccess_svhn(x):

    with torch.no_grad():
        x = Variable(torch.FloatTensor(x))

    x = x.transpose(0, 3)
    x = x.transpose(1, 2)
    x = x.transpose(2, 3)

    x /= 255

    return x


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


def gaussian_noise(X):
    X_copy = copy.deepcopy(X)
    X_copy = torch.FloatTensor(X_copy)

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans_noise(a)

    return X_copy


def color_jitter(X):
    X_copy = copy.deepcopy(X)
    X_copy = torch.FloatTensor(X_copy)

    for i in range(X_copy.shape[0]):
        a = F.to_pil_image(X_copy[i])
        X_copy[i] = trans_color_jitter(a)

    return X_copy


def just_scale(X, size, pad):
    X_copy = copy.deepcopy(X)
    X_copy = Variable(torch.FloatTensor(X_copy))

    j_scale = transforms.Resize(size=size, interpolation=2)
    trans = transforms.Compose([j_scale, transforms.Pad(pad), transforms.ToTensor()])

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
    p = torch.round(p)
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
        scaled_up = randcom_crop_upscale_22(image, 22, quarter, 22, 32, 32)
        sobeled = sobel_total(scaled_up, quarter)
        AA = fix_sobel(sobeled, quarter, image, size_x, size_y)

        return AA

    elif id == 10:
        scaled_up = randcom_crop_upscale_22(image, 22, quarter, 22, 32, 32)
        rev = torch.abs(1 - scaled_up)
        return rev

    elif id == 11:
        rot = no_jitter_rotate(image, -46)
        return rot

    elif id == 12:
        scaled_up = randcom_crop_upscale_20(image, 20, quarter, 20, 32, 32)
        return scaled_up

    elif id == 13:
        scaled_up = randcom_crop_upscale_22(image, 22, quarter, 22, 32, 32)

        return scaled_up

    elif id == 14:
        scaled_up = randcom_crop_upscale_26(image, 26, quarter, 26, 32, 32)
        return scaled_up

    elif id == 15:
        scaled_up = no_jitter_random_corpse(image, 22, quarter, 22, 32, 32)

        return scaled_up

    elif id == 16:
        scaled_up = randcom_crop_upscale_18(image, 18, quarter, 18, 32, 32)
        return scaled_up

    elif id == 17:
        r_erased = random_erease(image)
        return r_erased

    elif id == 18:
        noised = gaussian_noise(image)
        return noised

    print("Error in transformation of the image.")
    input()
    return image
