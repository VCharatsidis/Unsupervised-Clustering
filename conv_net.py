from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """

        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchNorm6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()
        self.maxpool6 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.linear = nn.Linear(512, 256)

        # self.layers = nn.Sequential(
        #     nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        #
        #     nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        #
        #     nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        #
        #     nn.Linear(25088, 256)
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

        out = self.conv1(x)
        out = self.batchNorm1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.batchNorm2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.batchNorm3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)

        out = self.conv4(out)
        out = self.batchNorm4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)

        out = self.conv5(out)
        out = self.batchNorm5(out)
        out = self.relu5(out)
        out = self.maxpool5(out)

        out = self.conv6(out)
        out = self.batchNorm6(out)
        out = self.relu6(out)
        out = self.maxpool6(out)

        out = out.view(out.shape[0], -1)

        out = self.linear(out)
        # out = x
        # for layer in self.layers:
        #     if isinstance(layer, nn.Linear):
        #         out = out.view(out.shape[0], -1)
        #
        #     out = layer.forward(out)


        return out