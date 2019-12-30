import torch.nn as nn


class Guesser(nn.Module):

    def __init__(self, hidden_dim=500, input_dim=10):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

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
