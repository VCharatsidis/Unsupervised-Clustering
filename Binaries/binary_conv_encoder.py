from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch



class BinConvEncoderNet(nn.Module):

    def __init__(self, n_channels, n_inputs, dp, classes):

        super(BinConvEncoderNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
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

        conv = self.conv(x)
        encoding = torch.flatten(conv, 1)

        return encoding, encoding, encoding