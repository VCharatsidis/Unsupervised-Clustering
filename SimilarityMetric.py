import torch
import torch.nn as nn
from conv_net import ConvNet
from discriminator import MLP


class SimilarityMetric(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ConvNet(1)
        self.encoder2 = ConvNet(1)

        input_dim = 256
        self.discriminator = MLP(input_dim)


    def forward(self, image_1, image_2):
        encoded_1 = self.encoder(image_1)
        encoded_2 = self.encoder(image_2)

        discriminator_input = torch.cat([encoded_1, encoded_2], 1)
        output = self.discriminator(discriminator_input)

        return output
