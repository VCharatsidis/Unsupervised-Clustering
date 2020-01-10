import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        input_dim = 28 * 28
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.std = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, input):
        """
        Perform forward pass of encoder.
        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h = self.fc1(input)
        h = self.relu(h)

        mean = self.mean(h)
        mean = self.softmax(mean)

        std = self.std(h)
        std = self.softmax(std)

        return mean, std




