import torch
import torch.nn as nn
import math
import torchsummary

from multiprocessing import freeze_support
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import time

from torchvision.transforms import InterpolationMode

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        reduced_dim = max(1, int(in_channels * se_ratio))

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish(),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EfficientNet(nn.Module):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, num_classes):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        expand_ratios = [1, 6, 6, 6, 6, 6, 6]

        channels = [int(x * width_coefficient) for x in channels]
        repeats = [int(math.ceil(x * depth_coefficient)) for x in repeats]

        features = [nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            Swish()
        )]

        for i in range(7):
            for j in range(repeats[i]):
                stride = strides[i] if j == 0 else 1
                in_channels = channels[i] if j == 0 else channels[i + 1]
                features.append(MBConv(in_channels, channels[i + 1], kernel_sizes[i], stride, expand_ratios[i], 0.25))

        features.extend([
            nn.Conv2d(channels[-2
                      ], channels[-1], 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        ])

        self.features = nn.Sequential(*features)

    def forward(self, x):
        return self.features(x)




mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms_train = transforms.Compose([
  transforms.RandomRotation(degrees=(-20, 20)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-8, 8)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


data_transforms_test_val = transforms.Compose([
 transforms.Resize((300, 300)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# loading dataset directly from torchvision as suggested in our paper and we split the dataset into train, val, and test
train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', transform=data_transforms_train,
                                                download=True)
val_dataset = torchvision.datasets.Flowers102(root='./data', split='val', transform=data_transforms_test_val,
                                              download=True)
test_dataset = torchvision.datasets.Flowers102(root='./data', split='test', transform=data_transforms_test_val,
                                               download=True)

# Create data loaders to load the data in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)  # removed num_workers=2


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)


def efficientnet_b0(num_classes=102):
    return EfficientNet(1.0, 1.0, 0.2, num_classes)
# Instantiate the custom model
custom_modal = efficientnet_b0()
device = set_device()
custom_resnet = custom_modal.to(device)

torchsummary.summary(model, input_size=(3, 224, 224), device=device.type)
