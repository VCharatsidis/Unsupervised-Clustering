from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch



class SupervisedClassifier(nn.Module):
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
        super(SupervisedClassifier, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),


            # nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            #
            # nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            #
            # nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            #
            # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            #
            # nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            #
            # nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            #
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            #
            # nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            #
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            #
            # nn.Conv2d(512, 725, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(725),
            # nn.ReLU(),
            #
            # nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            # nn.AvgPool2d(kernel_size=(1, 1), stride=1, padding=0),
        )

        self.embeding_linear = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
        )

        self.test_linear = nn.Sequential(
            nn.Linear(128, 10),
           # nn.Softmax(dim=1)
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
        embedding = self.embeding_linear(encoding)

        test_preds = self.test_linear(embedding)

        return embedding, test_preds