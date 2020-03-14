from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch


class OneNetEntropy(nn.Module):
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

        super(OneNetEntropy, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.Sigmoid()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.Sigmoid()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(512)
        self.relu4 = nn.Sigmoid()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.linear1 = nn.Linear(8212, 1000)
        self.relu5 = nn.Sigmoid()

        self.linear2 = nn.Linear(1000, 10)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, p1, p2):
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
        out_1 = self.maxpool1(out)
        # print("conv 1: ")
        # print(out_1.shape)

        out = self.conv2(out_1)
        out = self.batchNorm2(out)
        out = self.relu2(out)
        out_2 = self.maxpool2(out)
        # print("conv 2: ")
        # print(out_2.shape)

        out = self.conv3(out_2)
        out = self.batchNorm3(out)
        out = self.relu3(out)
        out_3 = self.maxpool3(out)
        # print("conv 3: ", out_3.shape)

        out = self.conv4(out_3)
        out = self.batchNorm4(out)
        out = self.relu4(out)
        out_4 = self.maxpool4(out)
        # print("conv 4: ", out_4.shape)

        out = torch.flatten(out_4, 1)
        # print("out", out.shape)
        # print("p1", p1.shape)
        # print("p2", p2.shape)
        out = torch.cat([out, p1, p2], 1)

        # print("before linear: ", out.shape)
        out = self.linear1(out)
        out_5 = self.relu5(out)

        out = self.linear2(out_5)
        out_6 = self.softmax(out)

        return out_1, out_2, out_3, out_4, out_5, out_6