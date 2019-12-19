"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch


class Colon(nn.Module):
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
        super(Colon, self).__init__()
        half = n_inputs // 2
        quarter = half // 2
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, half),
            nn.Tanh(),

            nn.Linear(half, half),
            nn.Tanh(),

            nn.Linear(half, quarter),
            nn.Tanh(),

            nn.Linear(quarter, 1),
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

        out = x
        for layer in self.layers:
            out = layer.forward(out)

        return out