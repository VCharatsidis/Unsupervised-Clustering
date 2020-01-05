import torch.nn as nn
from conv_net import ConvNet
from discriminator import MLP
import torch
import random


class MutualInfoMetric(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ConvNet(1)
        self.encoder.cuda()
        #self.encoder_2 = ConvNet(1)

        self.discriminator = MLP(12800)
        self.discriminator.cuda()

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

    def get_results(self, counter, results, encoded_1, encoded_2):
        for i in encoded_1:
            for j in encoded_2:
                results[counter] = self.pair_forward(i, j)
                counter += 1

        return counter, results

    def forward(self, set_1, set_2, set_3, set_4):

        results_encoder_set_1, results_encoder_set_2 = self.encode(set_1, set_2, self.encoder)
        results_encoder_set_3, results_encoder_set_4 = self.encode(set_3, set_4, self.encoder)

        combinations = len(results_encoder_set_1) * 32
        results = torch.zeros(combinations)

        counter = 0
        counter, results = self.get_results(counter, results, results_encoder_set_1, results_encoder_set_1)
        counter, results = self.get_results(counter, results, results_encoder_set_2, results_encoder_set_2)
        counter, results = self.get_results(counter, results, results_encoder_set_3, results_encoder_set_3)
        counter, results = self.get_results(counter, results, results_encoder_set_4, results_encoder_set_4)

        counter, results = self.get_results(counter, results, results_encoder_set_1, results_encoder_set_2)
        counter, results = self.get_results(counter, results, results_encoder_set_2, results_encoder_set_3)
        counter, results = self.get_results(counter, results, results_encoder_set_3, results_encoder_set_4)
        counter, results = self.get_results(counter, results, results_encoder_set_4, results_encoder_set_1)

        # for i in results_encoder_set_1:
        #     for j in results_encoder_set_1:
        #         results[counter] = self.pair_forward(i, j)
        #         counter += 1
        #
        # for i in results_encoder_set_2:
        #     for j in results_encoder_set_2:
        #         results[counter] = self.pair_forward(i, j)
        #         counter += 1
        #
        # for i in results_encoder_set_1:
        #     for j in results_encoder_set_2:
        #         results[counter] = self.pair_forward(i, j)
        #         counter += 1
        #
        # for i in results_encoder_set_2:
        #     for j in results_encoder_set_1:
        #         results[counter] = self.pair_forward(i, j)
        #         counter += 1


        return results




