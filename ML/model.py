import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, middle_layer):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, middle_layer[0])
        self.fc2 = nn.Linear(middle_layer[0], middle_layer[1])
        self.fc3 = nn.Linear(middle_layer[1], 3)

    def forward(self, x):
        h1 = self.fc1(x)
        z1 = torch.relu(h1)
        h2 = self.fc2(z1)
        z2 = torch.relu(h2)
        h3 = self.fc3(z2)
        y = torch.softmax(h3, dim=1)
        return y
