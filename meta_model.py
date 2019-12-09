"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn



class MetaMLP(nn.Module):
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

        super(MetaMLP, self).__init__()
        n_hidden = n_inputs
        hlaf = n_hidden // 2
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),

            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),

            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),

            nn.Linear(n_hidden, hlaf),
            nn.BatchNorm1d(hlaf),
            nn.ReLU(),

            nn.Linear(hlaf, hlaf),
            nn.BatchNorm1d(hlaf),
            nn.ReLU(),

            nn.Linear(hlaf, hlaf),
            nn.BatchNorm1d(hlaf),
            nn.ReLU(),

            nn.Linear(hlaf, hlaf),
            nn.BatchNorm1d(hlaf),
            nn.ReLU(),

            nn.Linear(hlaf, hlaf//2),
            nn.BatchNorm1d(hlaf//2),
            nn.ReLU(),

            nn.Linear(hlaf//2, 1),
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