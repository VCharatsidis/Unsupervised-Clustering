from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch



class ClassSpecialist(nn.Module):
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
        super(ClassSpecialist, self).__init__()
        stride = 1
        max_s = 2
        self.conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),
            #
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1)
        )

        self.linear = nn.Sequential(
            nn.Linear(n_inputs, 800),
            nn.Tanh(),

            # nn.Dropout(0.5),
            nn.Linear(800, 1),
            nn.Sigmoid()
        )

    def forward(self, x, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0):
        conv = self.conv(x)
        conv = torch.flatten(conv, 1)

        linear_input = torch.cat([conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0], 1)

        output = self.linear(linear_input)
        return output


class CombinedSpecialists(nn.Module):
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
        super(CombinedSpecialists, self).__init__()
        stride = 1
        max_s = 2

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=stride, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),
            #
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=max_s, padding=1),
        )

        self.specialist1 = ClassSpecialist(n_inputs)
        self.specialist2 = ClassSpecialist(n_inputs)
        self.specialist3 = ClassSpecialist(n_inputs)
        self.specialist4 = ClassSpecialist(n_inputs)
        self.specialist5 = ClassSpecialist(n_inputs)
        self.specialist6 = ClassSpecialist(n_inputs)
        self.specialist7 = ClassSpecialist(n_inputs)
        self.specialist8 = ClassSpecialist(n_inputs)
        self.specialist9 = ClassSpecialist(n_inputs)
        self.specialist10 = ClassSpecialist(n_inputs)


    def forward(self, x, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        conv = self.conv(x)

        pred_1 = self.specialist1(conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
        pred_2 = self.specialist2(conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
        pred_3 = self.specialist3(conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
        pred_4 = self.specialist4(conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
        pred_5 = self.specialist5(conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
        pred_6 = self.specialist6(conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
        pred_7 = self.specialist7(conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
        pred_8 = self.specialist8(conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
        pred_9 = self.specialist9(conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)
        pred_10 = self.specialist10(conv, p1, p2, p3, p4, p5, p6, p7, p8, p9, p0)

        return pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9, pred_10