from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch



class DetachedNet(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_channels, EMBEDING_SIZE):
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
        super(DetachedNet, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.brain_1 = nn.Sequential(
            nn.Linear(6400, EMBEDING_SIZE),
            nn.Sigmoid()
        )

        self.brain_2 = nn.Sequential(
            nn.Linear(4608, EMBEDING_SIZE),
            nn.Sigmoid()
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

        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)

        encoding_1 = torch.flatten(conv_3, 1)
        embedding_1 = self.brain_1(encoding_1)

        conv_4 = self.conv_4(conv_3)
        encoding_2 = torch.flatten(conv_4, 1)
        embedding_2 = self.brain_2(encoding_2)

        return encoding_1, encoding_2, embedding_1, embedding_2


