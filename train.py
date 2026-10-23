import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from models.cnn_models import ResidualCNN

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load EMNIST 'byclass' dataset
    train_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)

    # 1. Take a 10% random subset to speed up local CPU training
    subset_indices = np.random.choice(len(train_dataset), size=int(len(train_dataset) * 0.10), replace=False)
    train_subset = Subset(train_dataset, subset_indices)

    # 2. Increase batch size and enable multi-core data loading
    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True, num_workers=2)

    model = ResidualCNN(num_classes=62).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Reduce epochs to 1 or 2 for mini-project demo runs
    epochs = 2
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"--- Epoch [{epoch+1}/{epochs}] Complete. Avg Loss: {total_loss / len(train_loader):.4f} ---")

    # Save trained parameters
    torch.save(model.state_dict(), "resnet_emnist.pth")
    print("Model saved successfully to resnet_emnist.pth!")

if __name__ == '__main__':
    train()