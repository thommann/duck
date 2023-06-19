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


def krone_relu(x):
    x_a, x_b = x[0], x[1]
    h_a = torch.relu(x_a)
    h_b = torch.relu(x_b)
    return torch.stack([h_a, h_b], dim=0)


def krone_softmax(x):
    x_a, x_b = x[0], x[1]
    z = torch.kron(x_a, x_b)
    y = torch.softmax(z, dim=1)
    return y


class KroneLinear(nn.Module):
    def __init__(self, in_features_a, in_features_b, out_features_a, out_features_b):
        super(KroneLinear, self).__init__()
        self.in_features_a = in_features_a
        self.in_features_b = in_features_b
        self.out_features_a = out_features_a
        self.out_features_b = out_features_b
        self.weight_a = nn.Parameter(torch.Tensor(out_features_a, in_features_a))
        self.weight_b = nn.Parameter(torch.Tensor(out_features_b, in_features_b))
        self.bias_a = nn.Parameter(torch.Tensor(out_features_a))
        self.bias_b = nn.Parameter(torch.Tensor(out_features_b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_a, x_b = x[0], x[1]
        h_a = torch.matmul(x_a, self.weight_a.t()) + self.bias_a
        h_b = torch.matmul(x_b, self.weight_b.t()) + self.bias_b
        return torch.stack([h_a, h_b], dim=0)


class KroneNet(nn.Module):
    def __init__(self, middle_layer_a, middle_layer_b):
        super(KroneNet, self).__init__()
        self.fc1 = KroneLinear(2, 2, middle_layer_a[0], middle_layer_b[0])
        self.fc2 = KroneLinear(middle_layer_a[0], middle_layer_b[0], middle_layer_a[1], middle_layer_b[1])
        self.fc3 = KroneLinear(middle_layer_a[1], middle_layer_b[1], 1, 3)

    def forward(self, x):
        h1 = self.fc1(x)
        z1 = krone_relu(h1)
        h2 = self.fc2(z1)
        z2 = krone_relu(h2)
        h3 = self.fc3(z2)
        y = krone_softmax(h3)
        return y
