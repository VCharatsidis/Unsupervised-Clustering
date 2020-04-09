from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch



class OneNet(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_channels, n_inputs, number_classes):
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
        super(OneNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
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

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

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

        self.dropout = nn.Sequential(
            nn.Dropout2d(0.1)
        )

        self.linear_embedding = nn.Sequential(
            # nn.Linear(n_inputs, 600),
            # nn.Tanh(),
            #
            # nn.Linear(600, 300),
            # nn.Tanh(),

            nn.Linear(n_inputs, 128),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            # nn.Linear(n_inputs, 600),
            # nn.Tanh(),
            #
            # nn.Linear(600, 300),
            # nn.Tanh(),

            nn.Linear(128, number_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x, p1, p2, p3, p4, p5, mean):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        conv = self.conv(x)
        embeddings = torch.flatten(conv, 1)

        #droped = self.dropout(conv)

        droped_embeddings = torch.flatten(embeddings, 1)

        #batch_size = p1.shape[0]

        #m_pred = mean.expand(batch_size, 10)

        linear_input = torch.cat([droped_embeddings, p1, p2, p3, p4, p5], 1)
        lin_embedding = self.linear_embedding(linear_input)

        preds = self.linear(lin_embedding)

        return lin_embedding, preds