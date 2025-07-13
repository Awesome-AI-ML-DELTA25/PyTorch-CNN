import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random

def get_loaders(batch_size=64, val_split=0.1, seed=42,
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    transform = transforms.Compose([
        # transforms.Resize(224)
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load full train and test datasets
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    # Calculate sizes
    total_train = len(full_train_dataset)  # 50,000
    val_size = int(val_split * total_train)  # e.g. 5,000
    train_size = total_train - val_size     # e.g. 45,000

    # Split the full training set
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(seed))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, val_loader, test_loader
