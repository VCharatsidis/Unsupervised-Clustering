from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch



class MultiHeadBatch(nn.Module):

    def __init__(self, n_channels, n_inputs, dp, classes):

        super(MultiHeadBatch, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        # self.head_input = 128
        # self.embeding_linear = nn.Sequential(
        #     #nn.Dropout2d(0.2),
        #     nn.Linear(n_inputs, self.head_input),
        #     nn.ReLU(),
        # )

        self.headA = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(4608, classes[0])
        )

        self.headB = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(4608, classes[0])
        )

        # self.headC = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(4608, classes[0])
        # )
        #
        # self.headD = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(4608, classes[0])
        # )

        self.softmax = nn.Sequential(
            nn.Softmax(dim=1)
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

        conv = self.conv(x)
        encoding = torch.flatten(conv, 1)
        #embeddings = self.embeding_linear(encoding)

        test_preds10 = self.headA(encoding)
        probs10 = self.softmax(test_preds10)

        test_preds12 = self.headB(encoding)
        probs12 = self.softmax(test_preds12)

        # test_preds20 = self.headC(encoding)
        # probs20 = self.softmax(test_preds20)
        #
        # test_preds50 = self.headD(encoding)
        # probs50 = self.softmax(test_preds50)

        return encoding, probs10, probs12#, probs20, probs50