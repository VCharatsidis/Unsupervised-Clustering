from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch
import numpy as np
from operator import mul
import sys


class Brain(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_channels):
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
        super(Brain, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(),

            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),

            #nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            #nn.AvgPool2d(kernel_size=(1, 1), stride=1, padding=0),
        )

        self.eps = sys.float_info.epsilon

        self.colons = nn.ModuleList([
            nn.Sequential(
                nn.Linear(592, 600),
                nn.Tanh(),

                nn.Linear(600, 10),
                nn.Softmax(dim=1)
            )

            for _ in range(16)])


    def neighbors(slef, i, w, h, mode=8):
        """Return a list of neighbors.

        Works as like a 2d graph of 'w' width and 'h' height with boundaries.

        Args:
            i(int): 1d index
            w(int): width of the graph.
            h(int): height of the graph.
            mode(int): 8 for eight directions (includes diagonals); else for
                4 directions (top, down, left, right).

        Returns:
            list
        """

        size = w * h
        neighbors = []
        if i - w >= 0:
            neighbors.append(i - w)  # north
        if i % w != 0:
            neighbors.append(i - 1)  # west

        if (i + 1) % w != 0:
            neighbors.append(i + 1)  # east

        if i + w < size:
            neighbors.append(i + w)  # south

        if mode == 8:
            if ((i - w - 1) >= 0) and (i % w != 0):
                neighbors.append(i - w - 1)  # northwest

            if ((i - w + 1) >= 0) and ((i + 1) % w != 0):
                neighbors.append(i - w + 1)  # northeast

            if ((i + w - 1) < size) and (i % w != 0):
                neighbors.append(i + w - 1)  # southwest

            if ((i + w + 1) < size) and ((i + 1) % w != 0):
                neighbors.append(i + w + 1)  # southeast
        return neighbors


    def forward(self, x, train, optimizers):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """

        conv = self.conv(x)
        #print(conv.shape)

        conv = torch.flatten(conv, 2)
        #print(conv.shape)

        p1 = torch.zeros([conv.shape[0], 10])
        p2 = torch.zeros([conv.shape[0], 10])
        p3 = torch.zeros([conv.shape[0], 10])
        p4 = torch.zeros([conv.shape[0], 10])
        p5 = torch.zeros([conv.shape[0], 10])
        p6 = torch.zeros([conv.shape[0], 10])
        p7 = torch.zeros([conv.shape[0], 10])
        p8 = torch.zeros([conv.shape[0], 10])

        p1 = p1.to('cuda')
        p2 = p2.to('cuda')
        p3 = p3.to('cuda')
        p4 = p4.to('cuda')
        p5 = p5.to('cuda')
        p6 = p6.to('cuda')
        p7 = p7.to('cuda')
        p8 = p8.to('cuda')

        dim = conv.shape[2]
        predictions = [self.colons[i](torch.cat([p1, p2, p3, p4, p5, p6, p7, p8, conv[:, :, i]], 1)) for i in range(dim)]

        first_predictions = torch.zeros(predictions[0].shape)
        first_predictions = first_predictions.to('cuda')

        for p in predictions:
            first_predictions += p

        first_predictions /= len(predictions)

        product = first_predictions.mean(dim=0)
        log_product = torch.log(product)
        loss = - log_product.mean(dim=0)

        if train:
            torch.autograd.set_detect_anomaly(True)

            for i in optimizers:
                i.zero_grad()

            loss.backward(retain_graph=True)

            for i in optimizers:
                i.step()


        number_neighbours = 8
        second_guess = []
        for i in range(dim):
            n = self.neighbors(i, 4, 4, number_neighbours)

            neighbs = [predictions[nei] for nei in n]

            while len(neighbs) < number_neighbours:
                neighbs.append(p1)

            neighbs.append(conv[:, :, i])

            conc = torch.cat(neighbs, 1)
            second_guess.append(self.colons[i](conc))

        # print(predictions[0].shape)
        # print("predictions", len(predictions))

        sum_predictions = torch.zeros(second_guess[0].shape)
        sum_predictions = sum_predictions.to('cuda')

        for p in second_guess:

            sum_predictions += p

        sum_predictions /= len(second_guess)
        squared_sum_preds = sum_predictions * sum_predictions

        #product_predictions[(product_predictions < self.eps).data] = self.eps

        return squared_sum_preds, second_guess