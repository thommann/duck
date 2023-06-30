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


def krone_relu(x):
    x_a, x_b = x[0], x[1]
    h_a = torch.relu(x_a)
    h_b = torch.relu(x_b)
    return torch.stack([h_a, h_b], dim=0)


def krone_sigmoid(x):
    x_a, x_b = x[0], x[1]
    h_a = torch.sigmoid(x_a)
    h_b = torch.sigmoid(x_b)
    return torch.stack([h_a, h_b], dim=0)


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
        # Remove padding from previous layer
        x_b = x_b[:, :self.in_features_b]
        x_a = x_a[:, :self.in_features_a]
        # Compute output
        h_a = torch.matmul(x_a, self.weight_a.t()) + self.bias_a
        h_b = torch.matmul(x_b, self.weight_b.t()) + self.bias_b
        # Pad with zeros to make shapes match
        if self.out_features_a > self.out_features_b:
            h_b = torch.cat([h_b, torch.zeros(h_b.shape[0], self.out_features_a - self.out_features_b)], dim=1)
        elif self.out_features_b > self.out_features_a:
            h_a = torch.cat([h_a, torch.zeros(h_a.shape[0], self.out_features_b - self.out_features_a)], dim=1)
        return torch.stack((h_a, h_b), dim=0)


class KroneSoftmax(nn.Module):
    def __init__(self, in_features_a, in_features_b):
        super(KroneSoftmax, self).__init__()
        self.in_features_a = in_features_a
        self.in_features_b = in_features_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_a, x_b = x[0], x[1]
        # Remove padding from previous layer
        x_b = x_b[:, :self.in_features_b]
        x_a = x_a[:, :self.in_features_a]

        # Expand dimensions to perform broadcasting
        x_a_expanded = x_a[:, :, None, None]
        x_b_expanded = x_b[:, None, :, None]

        # Compute the Kronecker product for all samples (batch-wise) using broadcasting
        y = x_a_expanded * x_b_expanded

        # Reshape to final form
        batch_size = y.shape[0]
        y = y.reshape(batch_size, self.in_features_a * self.in_features_b)

        # Apply softmax
        y = torch.softmax(y, dim=1)

        return y


class KroneNet(nn.Module):
    def __init__(self, middle_layer_a, middle_layer_b, sigmoid=False):
        super(KroneNet, self).__init__()
        self.fc1 = KroneLinear(2, 2, middle_layer_a[0], middle_layer_b[0])
        self.fc2 = KroneLinear(middle_layer_a[0], middle_layer_b[0], middle_layer_a[1], middle_layer_b[1])
        self.fc3 = KroneLinear(middle_layer_a[1], middle_layer_b[1], 1, 3)
        self.softmax = KroneSoftmax(1, 3)
        self.activation = krone_sigmoid if sigmoid else krone_relu

    def forward(self, x):
        h1 = self.fc1(x)
        z1 = self.activation(h1)
        h2 = self.fc2(z1)
        z2 = self.activation(h2)
        h3 = self.fc3(z2)
        y = self.softmax(h3)
        return y
