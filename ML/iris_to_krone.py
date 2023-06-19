# Load the dataset
import torch
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ML.calculate_kronecker import calculate_kronecker
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

# Load iris model
state_dict = torch.load(f'data/iris-model{middle_layer[0]}x{middle_layer[1]}.pth')
krone_state_dict = {}
for key, value in state_dict.items():
    value_a, value_b = calculate_kronecker(value.numpy(), cc=True)
    print(key, value_a.shape, value_b.shape)
    tensor_a, tensor_b = torch.tensor(value_a), torch.tensor(value_b)
    # Squeeze the tensors if it is a bias
    if key.endswith("bias"):
        tensor_a = torch.squeeze(tensor_a, dim=1)
        tensor_b = torch.squeeze(tensor_b, dim=1)
    krone_state_dict[key + "_a"] = tensor_a
    krone_state_dict[key + "_b"] = tensor_b

middle_layer_a = [25, 20]
middle_layer_b = [40, 25]
model = KroneNet(middle_layer_a, middle_layer_b)
model.load_state_dict(krone_state_dict)

# Save the model
filename = f'data/iris-model{middle_layer[0]}x{middle_layer[1]}krone.pth'
torch.save(model.state_dict(), filename)
print(f"Model saved to {filename}")
