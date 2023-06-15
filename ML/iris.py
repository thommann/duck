import numpy as np
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch import optim

from ML.model import Net


def test(model: Net, x: torch.Tensor, y: torch.Tensor) -> None:
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    print(f"Accuracy: {100 * correct / total}%")


# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Scale the features for better training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.long)
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# Define the model
model = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1_000):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Test the model
print("Testing the model...")
test(model, X_test, y_test)

# Test the model on the whole dataset
print("Testing the model on the whole dataset...")
test(model, X, y)

# Save the model
state_dict = model.state_dict()
torch.save(state_dict, 'data/iris-model.pth')
for key, value in state_dict.items():
    value = np.atleast_2d(value)
    value = value.T
    size = value.shape
    print(key, size)
    np.savetxt(f"data/{key}_{size[0]}x{size[1]}.csv", value, delimiter=",")
