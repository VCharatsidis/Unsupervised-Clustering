from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
from Predictor import Predictor


class Encoder(nn.Module):
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

        super(Encoder, self).__init__()

        # self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
        # self.batchNorm1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.Tanh()
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        #
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.batchNorm2 = nn.BatchNorm2d(128)
        # self.relu2 = nn.Tanh()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        #
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.batchNorm3 = nn.BatchNorm2d(256)
        # self.relu3 = nn.Tanh()
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # self.conv4 = nn.Conv2d(64, 5, kernel_size=3, stride=1, padding=1)
        # self.batchNorm4 = nn.BatchNorm2d(5)
        # self.relu4 = nn.Tanh()
        # self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        #
        # self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.batchNorm5 = nn.BatchNorm2d(256)
        # self.relu5 = nn.Tanh()
        # self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        #
        # self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.batchNorm6 = nn.BatchNorm2d(256)
        # self.relu6 = nn.Tanh()
        # self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        #
        # self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # self.batchNorm7 = nn.BatchNorm2d(128)
        # self.relu7 = nn.Tanh()
        # self.maxpool7 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        #
        # self.conv8 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.batchNorm8 = nn.BatchNorm2d(64)
        # self.relu8 = nn.ReLU()
        # self.maxpool8 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        #
        # self.linear = nn.Linear(512, 20)

        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),

            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),

            # nn.Conv2d(256, 64, kernel_size=(3, 3), stride=1, padding=1),
            # #nn.BatchNorm2d(64),
            # nn.Tanh(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1),

            nn.Linear(1600, 100),
            nn.Tanh(),
            nn.Linear(100, 10),
            nn.Softmax()
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

        # out = self.conv1(x)
        # out = self.batchNorm1(out)
        # out = self.relu1(out)
        # out_1 = self.maxpool1(out)
        # # print("conv 1: ")
        # # print(out_1.shape)
        #
        # out = self.conv2(out_1)
        # out = self.batchNorm2(out)
        # out = self.relu2(out)
        # out_2 = self.maxpool2(out)
        # # print("conv 2: ")
        # # print(out_2.shape)
        #
        # out = self.conv3(out_2)
        # out = self.batchNorm3(out)
        # out = self.relu3(out)
        # out_3 = self.maxpool3(out)
        # print("conv 3: ")
        # print(out_3.shape)

        # out = self.conv4(out_3)
        # out = self.batchNorm4(out)
        # out = self.relu4(out)
        # out_4 = self.maxpool4(out)
        # print("conv 4: ")
        # print(out_4.shape)
        # print("conv 4 [0][0]")
        # print(out_4[0][0])
        #
        # out = self.conv5(out_4)
        # out = self.batchNorm5(out)
        # out = self.relu5(out)
        # out_5 = self.maxpool5(out)
        # print("conv 5: ")
        # print(out_5.shape)
        # print("conv 5 [0][0]")
        # print(out_5[0][0])
        #
        # out = self.conv5(out_5)
        # out = self.batchNorm5(out)
        # out = self.relu5(out)
        # out_6 = self.maxpool5(out)
        # print("conv 6: ")
        # print(out_6.shape)
        # print("conv 6 [0][0]")
        # print(out_6[0][0])
        # input()

        # out = self.conv6(out_5)
        # out = self.batchNorm6(out)
        # out = self.relu6(out)
        # out_6 = self.maxpool6(out)
        #
        # out = self.conv7(out_6)
        # out = self.batchNorm7(out)
        # out = self.relu7(out)
        # out_7 = self.maxpool7(out)
        #
        # out = self.conv8(out_7)
        # out = self.batchNorm8(out)
        # out = self.relu8(out)
        # out_8 = self.maxpool8(out)

        #
        # out = out.view(out.shape[0], -1)
        # out = self.linear(out)

        #out = self.linear(out)
        out = x
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                out = out.view(out.shape[0], -1)

            out = layer.forward(out)

        return out