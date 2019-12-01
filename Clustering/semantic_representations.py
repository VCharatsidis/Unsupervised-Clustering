import os
import torch
from datasets.bmnist import bmnist, my_bmnist
import torch
import matplotlib
from torchvision.utils import make_grid
import numpy as np
from train import _read_raw_image_file, noise_pixel
from torch.autograd import Variable


filepath = '..\\'
script_directory = os.path.split(os.path.abspath(__file__))[0]
encoder_model = os.path.join(script_directory, filepath + 'encoder.model')
encoder = torch.load(encoder_model)

decoder_model = os.path.join(script_directory, filepath + 'decoder.model')
decoder = torch.load(decoder_model)

X_train = _read_raw_image_file('..\\data\\raw\\binarized_mnist_train.amat')
X_test = _read_raw_image_file('..\\data\\raw\\binarized_mnist_valid.amat')

# for sample in train:
#     input = sample.reshape(sample.shape[0], -1)
#     print(input)


def display_reconstructions(x):
    sample = make_grid(x, nrow=1).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images\mnist.png", sample)


def get_semantic_representations():
    size = len(X_train)

    for id in range(size):

        with torch.no_grad():
            ids = []
            ids.append(id)
            X_test_batch = X_train[ids, ]
            #X_test_batch = [[[noise_pixel(pixel) for pixel in row] for row in image] for image in X_test_batch]
            X_test_batch = np.expand_dims(X_test_batch, axis=0)
            X_test_batch = Variable(torch.IntTensor(X_test_batch).float())

            encoder_output = encoder.forward(X_test_batch)
            encoded = encoder_output.detach().numpy()
            file = open("encoded_reps.txt", "a")
            for i in encoded[0]:
                file.writelines(str(i)+' ')

            file.writelines(str(id)+" \n")
            file.close()


get_semantic_representations()
# display_reconstructions(0)
# display_reconstructions(1)
# display_reconstructions(2)
# display_reconstructions(3)
# display_reconstructions(4)
# display_reconstructions(5)
# display_reconstructions(6)
# display_reconstructions(7)

