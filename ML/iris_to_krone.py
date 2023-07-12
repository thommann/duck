# Load the dataset
import torch
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ML.model import KroneNet
from ML.params import middle_layer

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Scale the features for better training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert numpy arrays to PyTorch tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# Save the model
filename = f'data/iris-model{middle_layer[0]}x{middle_layer[1]}_krone.pth'
torch.save(model.state_dict(), filename)
print(f"Model saved to {filename}")
