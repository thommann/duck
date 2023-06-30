import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

from ML.model import MnistNet
from ML.params import middle_layer, use_sigmoid


def test(trained_model: MnistNet, dataloader: DataLoader, device) -> None:
    # Test the model
    trained_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for test_images, test_labels in dataloader:
            test_images = test_images.view(test_images.shape[0], -1)  # flattening the image here
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            outputs = trained_model(test_images)
            _, predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
        print(f"Accuracy: {100 * correct / total}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Load and normalize the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=True, num_workers=4)

    # Define the model
    model = MnistNet(middle_layer, sigmoid=use_sigmoid).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Train the model
    for epoch in range(50):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)  # flattening the image here
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1} - Training loss: {running_loss / len(trainloader)}")

    # Test the model
    print("Testing the model...")
    test(model, testloader, device)

    # Save the model
    state_dict = model.state_dict()
    torch.save(state_dict, f'data/mnist-model{middle_layer[0]}x{middle_layer[1]}.pth')


if __name__ == '__main__':
    main()
