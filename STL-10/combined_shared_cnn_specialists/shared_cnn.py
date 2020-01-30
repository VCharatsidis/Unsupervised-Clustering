from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch



class CNNSpecialist(nn.Module):
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
        super(CNNSpecialist, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(n_inputs, 600),
            nn.Tanh(),

            # nn.Dropout(0.5),
            nn.Linear(600, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        output = self.linear(x)
        return output


class CombinedCNNSpecialists(nn.Module):
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
        super(CombinedCNNSpecialists, self).__init__()
        stride = 1
        max_s = 2

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=stride, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),
            #
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),
        )

        self.specialist1 = CNNSpecialist(n_inputs)
        self.specialist2 = CNNSpecialist(n_inputs)
        self.specialist3 = CNNSpecialist(n_inputs)
        self.specialist4 = CNNSpecialist(n_inputs)
        self.specialist5 = CNNSpecialist(n_inputs)
        self.specialist6 = CNNSpecialist(n_inputs)
        self.specialist7 = CNNSpecialist(n_inputs)
        self.specialist8 = CNNSpecialist(n_inputs)
        self.specialist9 = CNNSpecialist(n_inputs)
        self.specialist10 = CNNSpecialist(n_inputs)


    def forward(self, x):#, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        conv = self.conv(x)
        conv = torch.flatten(conv, 1)

        linear_input = conv # torch.cat([conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0], 1)

        pred_1 = self.specialist1(linear_input)
        pred_2 = self.specialist2(linear_input)
        pred_3 = self.specialist3(linear_input)
        pred_4 = self.specialist4(linear_input)
        pred_5 = self.specialist5(linear_input)
        pred_6 = self.specialist6(linear_input)
        pred_7 = self.specialist7(linear_input)
        pred_8 = self.specialist8(linear_input)
        pred_9 = self.specialist9(linear_input)
        pred_10 = self.specialist10(linear_input)

        return pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9, pred_10