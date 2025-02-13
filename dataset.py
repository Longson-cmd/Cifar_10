import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Define Image Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load CIFAR-10 Dataset
def get_dataloaders(batch_size=64):
    data_train = CIFAR10(root='my_dataset', train=True, transform=transform, download=True)
    data_test = CIFAR10(root='my_dataset', train=False, transform=transform, download=True)
    train_loader = DataLoader(data_train, shuffle=True, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(data_test, shuffle=False, batch_size=batch_size, num_workers=0)
    return train_loader, test_loader

