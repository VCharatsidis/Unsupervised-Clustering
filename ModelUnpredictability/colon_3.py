"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch


class Colon_3(nn.Module):
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
        super(Colon_3, self).__init__()

        self.base = nn.Linear(n_inputs, 3)
        self.tanh = nn.Tanh()

        self.a = nn.Linear(3, 1)
        self.b = nn.Linear(3, 1)
        self.c = nn.Linear(3, 1)

        self.sigmoid = nn.Sigmoid()

        # self.layers = nn.Sequential(
        #     nn.Linear(n_inputs, 3),
        #     nn.Tanh(),
        #
        #     nn.Linear(half, half),
        #     nn.Tanh(),
        #
        #     nn.Linear(half, quarter),
        #     nn.Tanh(),
        #
        #     nn.Linear(half, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        base = self.base(x)
        base = self.tanh(base)

        a = self.a(base)
        b = self.b(base)
        c = self.c(base)

        a = self.sigmoid(a)
        b = self.sigmoid(b)
        c = self.sigmoid(c)

        # out = x
        # for layer in self.layers:
        #     out = layer.forward(out)

        return a, b, c