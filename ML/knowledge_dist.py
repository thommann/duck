import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ML.calculate_kronecker import do_decomposition
from ML.model import KroneNet
from ML.params import middle_layer, middle_layer_a, middle_layer_b, use_sigmoid


def test(model: KroneNet, x: torch.Tensor, y: torch.Tensor) -> None:
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


def krone_input(X: torch.Tensor):
    X_krone_a = []
    X_krone_b = []
    for f_vec in X:
        f_vec_a, f_vec_b = do_decomposition(f_vec.numpy())
        f_vec_a, f_vec_b = torch.tensor(f_vec_a), torch.tensor(f_vec_b)
        f_vec_a = torch.squeeze(f_vec_a, dim=1)
        f_vec_b = torch.squeeze(f_vec_b, dim=1)
        X_krone_a.append(f_vec_a)
        X_krone_b.append(f_vec_b)

    X_krone = torch.stack((torch.stack(X_krone_a, dim=0), torch.stack(X_krone_b, dim=0)), dim=0)
    return X_krone


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

# Load iris model
model = KroneNet(middle_layer_a, middle_layer_b, sigmoid=use_sigmoid)
model.load_state_dict(torch.load(f'data/iris-model{middle_layer[0]}x{middle_layer[1]}krone.pth'))

# Calculate the Kronecker product of the input
X_krone = krone_input(X)
X_krone_train = krone_input(X_train)
X_krone_test = krone_input(X_test)

# Test the model on the whole dataset
print("Test raw model on the test dataset")
test(model, X_krone_test, y_test)
print("Test raw model on the whole dataset")
test(model, X_krone, y)

# Fine-tune the model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10_000):
    optimizer.zero_grad()
    output = model(X_krone_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} loss: {loss.item()}")
    if loss.item() < 0.6:
        break

print("Test fine-tuned model on the test dataset")
test(model, X_krone_test, y_test)
print("Test fine-tuned model on the whole dataset")
test(model, X_krone, y)
