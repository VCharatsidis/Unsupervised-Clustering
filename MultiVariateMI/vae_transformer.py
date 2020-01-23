import torch.nn as nn


class VaeDecoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
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


class VaeEncoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        input_dim = 28 * 28
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.std = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        """
        Perform forward pass of encoder.
        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h = self.fc1(input)
        h = self.relu(h)
        mean = self.mean(h)
        std = self.std(h)

        return mean, std
