import os
import torch
import matplotlib
from torchvision.utils import make_grid
import numpy as np
from torch.autograd import Variable
from sklearn.datasets import fetch_openml

filepath = '..\\'
script_directory = os.path.split(os.path.abspath(__file__))[0]
encoder_model = os.path.join(script_directory, filepath + 'encoder.model')
encoder = torch.load(encoder_model)

discriminator_model = os.path.join(script_directory, filepath + 'decoder.model')
discriminator = torch.load(discriminator_model)

mnist = fetch_openml('mnist_784', version=1, cache=True)
targets = mnist.target

X_train = mnist.data[:60000]
X_test = mnist.data[60000:]

# for sample in train:
#     input = sample.reshape(sample.shape[0], -1)
#     print(input)


def display_reconstructions(x):
    sample = make_grid(x, nrow=1).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images\mnist.png", sample)


def get_semantic_representations():
    size = len(X_train)
    threshold = 0

    for id in range(size):
        with torch.no_grad():
            ids = []
            ids.append(id)
            X_test_batch = X_train[ids, :]

            for i in range(X_test_batch.shape[0]):
                nums = np.random.uniform(low=0, high=1, size=(X_test_batch[i].shape[0],))
                X_test_batch[i] = np.where(nums > threshold, X_test_batch[i], 0)

            X_test_batch = np.reshape(X_test_batch, (1, 1, 28, 28))
            X_test_batch = Variable(torch.IntTensor(X_test_batch).float())

            encoder_output = encoder.forward(X_test_batch)
            encoded = encoder_output.detach().numpy()
            file = open("encoded_reps_30.txt", "a")
            for i in encoded[0]:
                file.writelines(str(i)+' ')

            file.writelines(str(id)+" \n")
            file.close()



def get_distances():
    with torch.no_grad():
        number_prototypes = 50
        prototypes_ids = np.random.choice(len(X_train), size=number_prototypes, replace=False)
        X_proto = X_train[prototypes_ids, :]

        size = len(X_train)

        X_proto = np.reshape(X_proto, (number_prototypes, 1, 28, 28))
        X_proto = Variable(torch.FloatTensor(X_proto))

        encoded_proto = encoder.forward(X_proto)

        for id in range(size):
            ids = [id]
            X_image = X_train[ids, :]
            X_image = np.reshape(X_image, (1, 1, 28, 28))
            X_image = Variable(torch.FloatTensor(X_image))

            encoded_image = encoder.forward(X_image)
            encoded_image = encoded_image.repeat(number_prototypes, 1)

            discriminator_input = torch.cat([encoded_image, encoded_proto], 1)
            discriminator_output = discriminator.forward(discriminator_input)

            disc = discriminator_output.detach().numpy()
            file = open("differences.txt", "a")
            for i in disc:

                file.writelines(str(i[0]) + ' ')

            file.writelines("\n")
            file.close()



#get_distances()
#get_semantic_representations()
# display_reconstructions(0)
# display_reconstructions(1)
# display_reconstructions(2)
# display_reconstructions(3)
# display_reconstructions(4)
# display_reconstructions(5)
# display_reconstructions(6)
# display_reconstructions(7)

