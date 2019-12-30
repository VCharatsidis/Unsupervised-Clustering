import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=10):
        super().__init__()

        input_dim = 28 * 28
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(z_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        """
        Perform forward pass of decoder.
        Returns mean with shape [batch_size, 784].
        """

        a = self.fc2(z)
        b = self.tanh(a)
        c = self.fc3(b)

        y = self.sigmoid(c)

        return y
