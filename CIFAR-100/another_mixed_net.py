from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch



class AnotherMixed(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_channels, EMBEDING_SIZE, classes):
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
        super(AnotherMixed, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        conv_size = 4096

        self.brain = nn.Sequential(

            nn.Linear(conv_size, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),

            nn.Linear(4096, EMBEDING_SIZE)
        )

        self.classification_conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(512),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.classification = nn.Sequential(

            # nn.Linear(conv_size, 4096),
            # nn.ReLU(),
            # nn.BatchNorm1d(4096),

            nn.Linear(6400, classes),
            nn.Softmax(dim=1)
        )

        self.sigmoid = nn.Sequential(
            nn.Sigmoid()
        )

        self.penalty_conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(512),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.penalty_classifier = nn.Sequential(

            nn.Linear(6400, classes),
            nn.Softmax(dim=1)
        )

        self.sigmoid = nn.Sequential(
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

        conv = self.conv(x)
        encoding = torch.flatten(conv, 1)

        class_conv = self.classification_conv(x)
        class_encoding = torch.flatten(class_conv, 1)
        classification = self.classification(class_encoding)

        pen_conv = self.penalty_conv(x)
        penalty_encoding = torch.flatten(pen_conv, 1)
        penalty_classification = self.penalty_classifier(penalty_encoding)

        logits = self.brain(encoding)
        binaries = self.sigmoid(logits)

        return encoding, classification, penalty_classification, binaries