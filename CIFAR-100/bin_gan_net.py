from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class BinGenerator(nn.Module):

    def __init__(self, n_channels, EMBEDING_SIZE):

        super(BinGenerator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.generator = nn.Sequential(
            # nn.Linear(4096, 4096),
            # nn.ReLU(),
            # nn.BatchNorm1d(4096),

            nn.Linear(4096, EMBEDING_SIZE)
        )

        self.sigmoid = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x):
        conv = self.conv(x)
        encoding = torch.flatten(conv, 1)

        logits = self.generator(encoding)
        binaries = self.sigmoid(logits)

        return encoding, logits, binaries


########################################################################


class Repelor(nn.Module):

    def __init__(self, n_input):

        super(Repelor, self).__init__()

        self.discriminator = nn.Sequential(
            # nn.Linear(n_input, 4096),
            # nn.ReLU(),
            # nn.BatchNorm1d(4096),

            nn.Linear(4096, 4096),
            nn.Sigmoid()
        )

    def forward(self, x):
        verdict = self.discriminator(x)

        return verdict


