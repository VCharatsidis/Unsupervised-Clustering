from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, z_dim=20):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.std = nn.Linear(hidden_dim, z_dim)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, input):
        """
        Perform forward pass of encoder.
        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h = self.fc1(input)
        h = self.relu(h)

        mean = self.mean(h)
        std = self.std(h)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, z_dim=20):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(z_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim-10)

    def forward(self, z):
        """
        Perform forward pass of decoder.
        Returns mean with shape [batch_size, 784].
        """
        a = self.fc2(z)
        b = self.tanh(a)
        c = self.fc3(b)

        y = self.sigmoid(c)

        return y


class OneNetVAE(nn.Module):

    def __init__(self, input_dim, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(input_dim, hidden_dim, z_dim).to("cuda")
        self.decoder = Decoder(input_dim, hidden_dim, z_dim).to("cuda")

    def forward(self, input, pred):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        flatten_image = torch.flatten(input, 1)
        concat = torch.cat([flatten_image, pred], 1)
        mean, std = self.encoder(concat)

        e = torch.zeros(mean.shape).normal_().to("cuda")
        z = std * e + mean

        output = self.decoder(z)

        return output


class OneNetGen(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs):
        """
        Initializes MLP object.
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
        """
        super(OneNetGen, self).__init__()


        self.linear = nn.Sequential(

            nn.Linear(n_inputs, 500),
            nn.Tanh(),
            #
            nn.Linear(500, 100),
            nn.Tanh(),

            nn.Linear(100, 500),
            nn.Tanh(),

            nn.Linear(500, n_inputs-10),
            nn.Sigmoid()
        )

    def forward(self, x, preds):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        flatten_image = torch.flatten(x, 1)
        concat = torch.cat([flatten_image, preds], 1)

        noised_image = self.linear(concat)

        return noised_image





