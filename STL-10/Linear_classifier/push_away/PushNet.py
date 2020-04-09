from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch



class PushNet(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_channels, n_inputs, dp, classes):
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
        super(PushNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            # nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            # nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),



            # nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            # nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            # nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            # nn.ReLU(),
            #
            # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            # nn.ReLU(),
            #
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            # nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            # nn.ReLU(),
            #
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            # nn.ReLU(),
            #
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            # nn.ReLU(),
            #
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            # nn.ReLU(),

            #nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #nn.AvgPool2d(kernel_size=(1, 1), stride=1, padding=0),
        )
        self.T = 1
        self.head_input = 128
        self.embeding_linear = nn.Sequential(
            nn.Linear(n_inputs, self.head_input),
            nn.ReLU(),
        )

        self.test_linear = nn.Sequential(
            nn.Linear(self.head_input, classes[0]),
        )

        self.help_linear1 = nn.Sequential(
            nn.Linear(self.head_input, classes[1]),
        )

        self.help_linear2 = nn.Sequential(
            nn.Linear(self.head_input, classes[2]),
        )

        self.help_linear3 = nn.Sequential(
            nn.Linear(self.head_input, classes[3]),
        )

        self.help_linear4 = nn.Sequential(
            nn.Linear(self.head_input, classes[4]),
        )

        self.softmax = nn.Sequential(
            nn.Softmax(dim=1)
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
        embeddings = self.embeding_linear(encoding)

        T = 0.1
        test_preds = self.softmax(self.test_linear(embeddings)/ T)

        help_preds1 = self.softmax(self.help_linear1(embeddings)/ T)
        help_preds2 = self.softmax(self.help_linear2(embeddings)/ T)
        help_preds3 = self.softmax(self.help_linear3(embeddings)/ T)
        help_preds4 = self.softmax(self.help_linear4(embeddings)/ T)



        return embeddings, test_preds, help_preds1, help_preds2, help_preds3, help_preds4