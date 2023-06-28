import numpy
import torch

weights = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
bias = numpy.array([1, 2, 3])
x = numpy.array([[1, 2, 3], [4, 5, 6]])
print(weights.shape)
print(bias.shape)
print(x.shape)
print(numpy.matmul(x, weights.T))
print(numpy.matmul(x, weights.T) + bias)

weights = torch.tensor(weights)
bias = torch.tensor(bias)
x = torch.tensor(x)
print(weights.shape)
print(bias.shape)
print(x.shape)
print(torch.matmul(x, weights.T))
print(torch.matmul(x, weights.T) + bias)

