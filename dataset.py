import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision.datasets import FashionMNIST

transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

train_dataset = FashionMNIST(root='data', train=True, download=True, transform=transform)
test_dataset = FashionMNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

image, label = train_dataset.__getitem__(0)
# print(image.shape)
# print(label)
# print(train_dataset.classes)

# DEBUG:
# for i, j in train_loader:
#     print(i.shape)
#     print(j)
#     break









