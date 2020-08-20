from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch



class SVHNComiteeEncoder(nn.Module):
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
        super(SVHNComiteeEncoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.embeding_linear1 = nn.Sequential(
            nn.Linear(3072, classes[0]),

        )

        self.embeding_linear2 = nn.Sequential(
            nn.Linear(3072, classes[0]),

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

        splited = torch.split(x, 16, dim=2)

        conv1 = self.conv(splited[0])
        conv2 = self.conv(splited[1])

        flat1 = torch.flatten(conv1, 1)
        embeddings1 = self.embeding_linear1(flat1)

        flat2 = torch.flatten(conv2, 1)
        embeddings2 = self.embeding_linear2(flat2)

        probs1 = self.softmax(embeddings1)
        probs2 = self.softmax(embeddings2)

        mean_prob = probs1 * probs2

        cat_emb = torch.cat([embeddings1, embeddings2], 1)

        return torch.cat([flat1, flat2], dim=1), cat_emb, mean_prob