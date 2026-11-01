import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import gc
from models.cnn_models import SimpleCNN, ResidualCNN

def train_model(model_type="simple"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training {model_type.upper()} CNN on device: {device} ---")

    # Limit CPU threads to prevent memory thrashing
    torch.set_num_threads(4)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)

    # 10% subset for fast local execution
    subset_indices = np.random.choice(len(train_dataset), size=int(len(train_dataset) * 0.10), replace=False)
    train_subset = Subset(train_dataset, subset_indices)
    
    # Reduced batch size to 128 to prevent RAM overflow on CPU
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)

    if model_type.lower() == "simple":
        model = SimpleCNN(num_classes=62).to(device)
        save_filename = "simple_cnn_emnist.pth"
    else:
        model = ResidualCNN(num_classes=62).to(device)
        save_filename = "resnet_emnist.pth"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"--- Epoch [{epoch+1}/{epochs}] Complete. Avg Loss: {total_loss / len(train_loader):.4f} ---")

    torch.save(model.state_dict(), save_filename)
    print(f"Model successfully saved to '{save_filename}'!\n")

    # Clean memory explicitly
    del model, optimizer, train_loader, train_dataset
    gc.collect()

if __name__ == '__main__':
    # Simple CNN is already trained, but you can run just residual now:
    train_model(model_type="residual")