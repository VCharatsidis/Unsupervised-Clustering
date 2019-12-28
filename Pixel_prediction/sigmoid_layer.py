from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class SigmoidLayer(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self):
        """
        Initializes ConvNet object.
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        TODO:
        Implement initialization of the network.
        """

        super(SigmoidLayer, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        out = self.sigmoid(x)

        return out