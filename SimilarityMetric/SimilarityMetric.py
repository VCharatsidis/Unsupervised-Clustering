import torch
import torch.nn as nn
from SimilarityMetric.conv_net import ConvNet
from SimilarityMetric.meta_model import MetaMLP


class SimilarityMetric(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ConvNet(1)
        self.encoder2 = ConvNet(1)

        input_dim = 512

        self.discriminator = MetaMLP(512)


    def flatten(self, out):
        return out.view(out.shape[0], -1)


    def forward(self, image_1, image_2):
        out_6 = self.encoder(image_1)
        out_2_6 = self.encoder2(image_2)

        out_inverse = self.encoder(image_2)
        out_inverse_2 = self.encoder2(image_1)

        # out_1 = self.flatten(out_1)
        # out_2_1 = self.flatten(out_2_1)
        #
        # out_2 = self.flatten(out_2)
        # out_2_2 = self.flatten(out_2_2)

        # out_3 = self.flatten(out_3)
        # out_2_3 = self.flatten(out_2_3)

        # out_4 = self.flatten(out_4)
        # out_2_4 = self.flatten(out_2_4)
        #
        # out_5 = self.flatten(out_5)
        # out_2_5 = self.flatten(out_2_5)

        out_6 = self.flatten(out_6)
        out_2_6 = self.flatten(out_2_6)

        out_inverse = self.flatten(out_inverse)
        out_inverse_2 = self.flatten(out_inverse_2)


        #discriminator_1_input = torch.cat([out_1, out_2_1], 1)
        #discriminator_2_input = torch.cat([out_2, out_2_2], 1)
        #discriminator_3_input = torch.cat([out_3, out_2_3], 1)
        # discriminator_4_input = torch.cat([out_4, out_2_4], 1)
        # discriminator_5_input = torch.cat([out_5, out_2_5], 1)
        discriminator_6_input = torch.cat([out_6, out_2_6], 1)
        inverse_input = torch.cat([out_inverse, out_inverse_2], 1)

        # print("disc")
        # # print(discriminator_1_input.shape)
        # # print(discriminator_2_input.shape)
        # print(discriminator_3_input.shape)
        # print(discriminator_4_input.shape)
        # print(discriminator_5_input.shape)
        # print(discriminator_6_input.shape)

        # output_1 = self.discriminator_1(discriminator_1_input)
        # output_2 = self.discriminator_2(discriminator_2_input)
        # output_3 = self.discriminator_3(discriminator_3_input)
        # output_4 = self.discriminator_4(discriminator_4_input.detach())
        # output_5 = self.discriminator_5(discriminator_5_input.detach())
        # output_6 = self.discriminator_6(discriminator_6_input)


        #print(meta_model_input.shape)

        output = self.discriminator(discriminator_6_input)
        output_inv = self.discriminator(inverse_input)

        return output, output_inv
