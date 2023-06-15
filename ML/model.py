import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 3)

    def forward(self, x):
        h1 = self.fc1(x)
        z1 = torch.relu(h1)
        h2 = self.fc2(z1)
        z2 = torch.relu(h2)
        h3 = self.fc3(z2)
        y = torch.softmax(h3, dim=1)
        return y
