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

        self.discriminator = MLP(20)
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

    def get_KLs(self, KLs, encoded_1, encoded_2):
        counter = 0
        for i in encoded_1:
            for j in encoded_2:
                print(self.kl_divergence(i, j))
                input()
                KLs[counter] = self.kl_divergence(i, j)

                counter += 1

        return KLs

    def kl_divergence(self, p, q):
        print(p)
        print(q)
        return torch.nn.functional.kl_div(p, q)


    def forward(self, set_1, set_2):

        results_encoder_set_1, results_encoder_set_2 = self.encode(set_1, set_2, self.encoder)
        #results_encoder_set_3, results_encoder_set_4 = self.encode(set_3, set_4, self.encoder)

        comparisons = 4
        versions = 7

        combinations = versions * versions * comparisons
        results = torch.zeros(combinations)

        counter = 0
        counter, results = self.get_results(counter, results, results_encoder_set_1, results_encoder_set_1)
        counter, results = self.get_results(counter, results, results_encoder_set_2, results_encoder_set_2)
        # counter, results = self.get_results(counter, results, results_encoder_set_3, results_encoder_set_3)
        # counter, results = self.get_results(counter, results, results_encoder_set_4, results_encoder_set_4)

        counter, results = self.get_results(counter, results, results_encoder_set_1, results_encoder_set_2)
        counter, results = self.get_results(counter, results, results_encoder_set_2, results_encoder_set_1)
        # counter, results = self.get_results(counter, results, results_encoder_set_2, results_encoder_set_3)
        # counter, results = self.get_results(counter, results, results_encoder_set_3, results_encoder_set_4)
        # counter, results = self.get_results(counter, results, results_encoder_set_4, results_encoder_set_1)

        KL_1 = torch.zeros(versions * versions)
        KL_2 = torch.zeros(versions * versions)

        KL_1 = self.get_KLs(KL_1, results_encoder_set_1, results_encoder_set_1)
        KL_2 = self.get_KLs(KL_2, results_encoder_set_2, results_encoder_set_2)

        return results, KL_1, KL_2




