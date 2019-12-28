import torch
import torch.nn as nn
import math
from encoder import Encoder
from decoder import Decoder
import matplotlib.pyplot as plt


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        mean, std = self.encoder(input)

        e = torch.zeros(mean.shape).normal_()
        z = std * e + mean

        output = self.decoder(z)

        eps = 1e-8
        L_reconstruction = self.calc_distance(output, input)

        KLD = 0.5 * (std.pow(2) + mean.pow(2) - 1 - torch.log(std.pow(2)+eps))

        elbo = KLD.sum(dim=-1) - L_reconstruction.sum(dim=-1)
        elbo = elbo.mean()

        if math.isnan(elbo) or math.isinf(elbo):
            print("elbo", elbo)
            pixels = output.detach().numpy().reshape((28, 28))
            plt.imshow(pixels, cmap='gray')
            plt.show()

            pixels = input.detach().numpy().reshape((28, 28))
            plt.imshow(pixels, cmap='gray')
            plt.show()

            abs_difference = torch.abs(output - input)
            print("difference ",abs_difference)
            information_loss = torch.log(1 - abs_difference)
            print("info loss",information_loss)
            print("output", output)
            print("input", input)

            input()

        return elbo

    def calc_distance(self, out, y):

        abs_difference = torch.abs(out - y)

        eps = 1e-8
        information_loss = torch.log(1 - abs_difference+eps)

        return information_loss

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        size = 28
        samples = torch.randn((n_samples, self.z_dim))
        y = self.decoder(samples)

        im_means = y.reshape(n_samples, 1, size, size)
        #sampled_ims = torch.bernoulli(im_means)

        return im_means

    def manifold_sample(self, n_samples):
        n = int(math.sqrt(n_samples))
        xy = torch.zeros(n_samples, 2)
        xy[:, 0] = torch.arange(0.01, n, 1 / n) % 1
        xy[:, 1] = (torch.arange(0.01, n_samples, 1) / n).float() / n
        z = torch.erfinv(2 * xy - 1) * math.sqrt(2)

        with torch.no_grad():
            mean = self.decoder(z)

        return mean
