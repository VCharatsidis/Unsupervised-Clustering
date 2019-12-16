import torch.nn as nn
from conv_net import ConvNet
from discriminator import MLP
import torch
import random


class MutualInfoMetric(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ConvNet(1)
        self.encoder_2 = ConvNet(1)

        self.discriminator = MLP(90)

    def flatten(self, out):
        return out.view(out.shape[0], -1)

    def pair_forward(self, encoded_image, encoded_image_2):
        mlp_in = torch.cat([encoded_image, encoded_image_2], 1)
        mlp_out = self.discriminator(mlp_in)

        return mlp_out

    def encode(self, set_1, set_2, encoder):
        encodings = []
        encodings_2 = []
        for image in set_1:
            out = encoder(image)
            out_flattened = self.flatten(out)
            encodings.append(out_flattened)

        for image in set_2:
            out = encoder(image)
            out_flattened = self.flatten(out)
            encodings_2.append(out_flattened)

        return encodings, encodings_2

    def forward(self, set_1, set_2):
        results = []

        results_encoder_set_1, results_encoder_set_2 = self.encode(set_1, set_2, self.encoder)
        results_encoder_2_set_1, results_encoder_2_set_2 = self.encode(set_1, set_2, self.encoder_2)

        for i in results_encoder_set_1:
            for j in results_encoder_set_2:
                res = self.pair_forward(i, j)
                results.append(res)

            for j in results_encoder_2_set_2:
                res = self.pair_forward(i, j)
                results.append(res)

        for i in results_encoder_2_set_1:
            for j in results_encoder_set_2:
                res = self.pair_forward(i, j)
                results.append(res)

            for j in results_encoder_2_set_2:
                res = self.pair_forward(i, j)
                results.append(res)

        return results




