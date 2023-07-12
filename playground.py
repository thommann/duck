import numpy
import torch

x = torch.tensor(numpy.array([[[1], [2], [3], [4]]]))
print(x.shape)
print(x)

reshape = x.view(-1, 2, 2).permute(0, 2, 1)
print(reshape.shape)
print(reshape)

vector = reshape.permute(0, 2, 1).reshape(-1, 4)
print(vector.shape)
print(vector)
