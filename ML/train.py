import numpy as np
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch import optim

from ML.model import IrisNet, MnistNet
from ML.params import middle_layer, use_sigmoid


def test(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> None:
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    print(f"Accuracy: {100 * correct / total:.1f}%")


def train(task: str):
    # Load the dataset
    if task == "iris":
        dataset = datasets.load_iris()
        model = IrisNet(middle_layer, sigmoid=use_sigmoid)
    elif task == "mnist":
        dataset = datasets.fetch_openml('mnist_784', version=1, cache=True, as_frame=False, parser='liac-arff')
        model = MnistNet(middle_layer, sigmoid=use_sigmoid)
    else:
        raise ValueError(f"Unknown task: {task}")

    X = dataset.data
    y = dataset.target
    y = y.astype(int)

    # Scale the features for better training
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Train the model
    for step in range(10_000):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step: {step}, Loss: {loss.item()}")
        if task == 'iris' and loss.item() < 0.6:
            break
        if task == 'mnist' and loss.item() < 1.6:
            break

    # Test the model
    print("Testing the model...")
    test(model, X_test, y_test)

    # Test the model on the whole dataset
    print("Testing the model on the whole dataset...")
    test(model, X, y)

    # Save the model
    state_dict = model.state_dict()
    torch.save(state_dict, f'data/{task}-model{middle_layer[0]}x{middle_layer[1]}.pth')


if __name__ == "__main__":
    train("iris")
    # train("mnist")
