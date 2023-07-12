import torch
from torch import nn


class IrisNet(nn.Module):
    def __init__(self, middle_layer, sigmoid=False):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, middle_layer[0])
        self.fc2 = nn.Linear(middle_layer[0], middle_layer[1])
        self.fc3 = nn.Linear(middle_layer[1], 3)
        self.activation = torch.sigmoid if sigmoid else torch.relu

    def forward(self, x):
        h1 = self.fc1(x)
        z1 = self.activation(h1)
        h2 = self.fc2(z1)
        z2 = self.activation(h2)
        h3 = self.fc3(z2)
        y = torch.softmax(h3, dim=1)
        return y


class MnistNet(nn.Module):
    def __init__(self, middle_layer, sigmoid=False):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, middle_layer[0])
        self.fc2 = nn.Linear(middle_layer[0], middle_layer[1])
        self.fc3 = nn.Linear(middle_layer[1], 10)
        self.activation = torch.sigmoid if sigmoid else torch.relu

    def forward(self, x):
        h1 = self.fc1(x)
        z1 = self.activation(h1)
        h2 = self.fc2(z1)
        z2 = self.activation(h2)
        h3 = self.fc3(z2)
        y = torch.softmax(h3, dim=1)
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
        self.bias = nn.Parameter(torch.Tensor(out_features_a * out_features_b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input to out_features_b x out_features_a
        vx = x.view(-1, self.in_features_b, self.in_features_a).permute(0, 2, 1)
        # Multiply with weight matrices
        bvx = torch.matmul(self.weight_b, vx)
        bvxa = torch.matmul(bvx, self.weight_a.t())
        # reshape to 2D
        vbvxa = bvxa.permute(0, 2, 1).reshape(-1, self.out_features_a * self.out_features_b)
        # Add bias
        y = vbvxa + self.bias
        return y


class KroneNet(nn.Module):
    def __init__(self, middle_layer_a, middle_layer_b, sigmoid=False):
        super(KroneNet, self).__init__()
        self.fc1 = KroneLinear(2, 2, middle_layer_a[0], middle_layer_b[0])
        self.fc2 = KroneLinear(middle_layer_a[0], middle_layer_b[0], middle_layer_a[1], middle_layer_b[1])
        self.fc3 = KroneLinear(middle_layer_a[1], middle_layer_b[1], 1, 3)
        self.activation = torch.sigmoid if sigmoid else torch.relu

    def forward(self, x):
        h1 = self.fc1(x)
        z1 = self.activation(h1)
        h2 = self.fc2(z1)
        z2 = self.activation(h2)
        h3 = self.fc3(z2)
        y = torch.softmax(h3, dim=1)
        return y
