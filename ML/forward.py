import torch
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ML.model import Net

# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Scale the features for better training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert numpy arrays to PyTorch tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# Load iris model
model = Net()
model.load_state_dict(torch.load('data/iris-model.pth'))

# # Test the model on the whole dataset
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     outputs = model(X)
#     _, predicted = torch.max(outputs.data, 1)
#     total += y.size(0)
#     correct += (predicted == y).sum().item()
#
# print(f"Accuracy of the model: {100 * correct / total}%")

# first input and target
x = torch.atleast_2d(X[0])
print(x)
y_0 = torch.atleast_1d(y[0])
print(y_0)

# Test the model on the first input
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(x)
    print("Output:", outputs.data)
    _, predicted = torch.max(outputs.data, 1)
    total += 1
    correct += (predicted == y_0).sum().item()

print(f"Accuracy of the model on the first input: {100 * correct / total}%")
