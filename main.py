import torch
import torch.nn as nn
import time
import os

from scripts.dataloader import get_loaders
from scripts.train import train
from scripts.AlexNet import AlexNet

if __name__ == '__main__':
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Create folders for results and weights
    os.makedirs("wts", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load data with train/val/test splits
    train_loader, val_loader, test_loader = get_loaders(batch_size=128)

    # Initialize the AlexNet model
    model = AlexNet(num_classes=10).to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Start training
    total_start_time = time.time()
    train(model, device, train_loader, val_loader, test_loader, criterion, optimizer, epochs=15)
    total_time = time.time() - total_start_time

    print(f"\n✅ Finished Training. Total Time: {total_time:.2f} seconds")

    # Save trained model weights
    model_path = "wts/alexnet_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model weights saved to: {model_path}")
