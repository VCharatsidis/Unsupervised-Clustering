from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch


class ColonMLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_channels, n_inputs):
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
        super(ColonMLP, self).__init__()

        self.linear = nn.Sequential(

            nn.Linear(n_inputs, 500),
            nn.Tanh(),


            nn.Linear(500, 10)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        logits = self.linear(x)
        out = self.softmax(logits)

        return out