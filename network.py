import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = transforms.Compose([
    # Resize to 256x256 then crop to 224x224
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(torch.tensor(mean), torch.tensor(std))
])

# loading dataset directly from torchvision as suggested in our paper and we split the dataset into train, val, and test
train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', transform=data_transforms, download=True)
val_dataset = torchvision.datasets.Flowers102(root='./data', split='val', transform=data_transforms, download=True)
test_dataset = torchvision.datasets.Flowers102(root='./data', split='test', transform=data_transforms, download=True)

# Create data loaders to load the data in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)