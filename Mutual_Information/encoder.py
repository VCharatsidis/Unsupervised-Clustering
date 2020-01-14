from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class Encoder(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """

        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),
        )

        self.linear = nn.Sequential(
            nn.Linear(6400, 100),
            nn.Tanh(),
            nn.Linear(100, 10),
            nn.Softmax()
        )

        self.transposed_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.tanh_1 = nn.Tanh()

        self.transposed_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2)
        self.tanh_2 = nn.Tanh()

        self.transposed_3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2)
        self.tanh_3 = nn.Sigmoid()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=3),
            nn.Tanh(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=3),
            nn.Tanh(),

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=1, padding=3)
        )


    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        """

        convolved = self.encoder(x)

        # print("convolved: ", convolved.shape)
        # out = self.transposed_1(convolved)
        # print("transpose 1: ", out.shape)
        # out = self.tanh_1(out)
        #
        # out = self.transposed_2(out)
        # print("transpose 2: ", out.shape)
        # out = self.tanh_2(out)
        #
        # out = self.transposed_3(out)
        # print("transpose 3: ", out.shape)
        # out = self.tanh_3(out)


        #decoding = self.decoder(convolved)

        convolved = self.linear(convolved.view(convolved.shape[0], -1))

        # out = x
        # for layer in self.encoder:
        #     if isinstance(layer, nn.Linear):
        #         out = out.view(out.shape[0], -1)
        #
        #     out = layer.forward(out)

        return convolved#, decoding