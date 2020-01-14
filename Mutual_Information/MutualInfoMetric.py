import torch.nn as nn
from discriminator import Discriminator
from encoder import Encoder

import torch


class MutualInfoMetric(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(1)
        self.encoder.cuda()

        self.discriminator = Discriminator(20)
        self.discriminator.cuda()

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.encoder.decoder.parameters(), lr=1e-4)

    def flatten(self, out):
        return out.view(out.shape[0], -1)

    def pair_forward(self, encoded_image, encoded_image_2):
        mlp_in = torch.cat([encoded_image, encoded_image_2], 1)
        mlp_out = self.discriminator(mlp_in)

        return mlp_out

    def encode(self, set_1, set_2, encoder):
        encodings = []
        encodings_2 = []

        decs_1 = []
        decs_2 = []
        for image in set_1:
            enc = encoder(image)
            out_flattened = self.flatten(enc)
            encodings.append(out_flattened)
            #decs_1.append(dec)

        for image in set_2:
            enc = encoder(image)
            out_flattened = self.flatten(enc)
            encodings_2.append(out_flattened)
            #decs_2.append(dec)

        return encodings, encodings_2, decs_1, decs_2

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
                softmax = torch.nn.functional.softmax(i, dim=1)
                softmax_2 = torch.nn.functional.softmax(j, dim=1)

                KLs[counter] = self.kl_divergence(softmax, softmax_2)

                counter += 1

        return KLs

    def kl_divergence(self, p, q):
        return torch.nn.functional.kl_div(torch.log(p), q)


    def forward(self, set_1, set_2):

        results_encoder_set_1, results_encoder_set_2, decs_1, decs_2 = self.encode(set_1, set_2, self.encoder)

        # for i in set_1:
        #     for j in decs_1:
        #         loss = self.criterion(j, i)
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()

        comparisons = 4
        versions = 7

        combinations = versions * versions * comparisons
        results = torch.zeros(combinations)

        counter = 0
        counter, results = self.get_results(counter, results, results_encoder_set_1, results_encoder_set_1)
        counter, results = self.get_results(counter, results, results_encoder_set_2, results_encoder_set_2)

        counter, results = self.get_results(counter, results, results_encoder_set_1, results_encoder_set_2)
        counter, results = self.get_results(counter, results, results_encoder_set_2, results_encoder_set_1)

        KL_1 = torch.zeros(versions * versions)
        KL_2 = torch.zeros(versions * versions)

        KL_1 = self.get_KLs(KL_1, results_encoder_set_1, results_encoder_set_1)
        KL_2 = self.get_KLs(KL_2, results_encoder_set_2, results_encoder_set_2)

        return results, KL_1, KL_2, decs_1, decs_2
