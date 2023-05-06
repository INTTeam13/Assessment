import torch
import torch.nn as nn
import math

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
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
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


# Add a function to validate the model on the validation dataset
def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    device = set_device()

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    print("Validation dataset - Classified %d out of %d images correctly (%.3f%%)" % (correct, total, accuracy))
    return accuracy


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)


def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    with torch.no_grad():  # Speeds up process, does not allow back-prop
        for data in test_loader:
            # Specifies batch size incase last batch does not end on even multiple of batch size
            # (e.g. 29 images in last batch not 32)
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)  # Models classification of the images

            _, predicted = torch.max(outputs.data, 1)  # 1 specifies dimension it is reduced to

            predicted_correctly_on_epoch += (labels == predicted).sum().item()

    epoch_accuracy = (predicted_correctly_on_epoch / total) * 100  # Get accuracy as percentage

    print("Test dataset - Classified %d out of %d images correctly (%.3f%%)" %
          (predicted_correctly_on_epoch, total, epoch_accuracy))


# Modify the train_network function to include validation and model saving
# Modify the train_network function to include learning rate scheduling
# Modify the train_network function to include learning rate scheduling
def train_network(model, train_loader, val_loader, test_loader, loss_function, optimizer, scheduler, n_epochs):
    device = set_device()
    best_val_accuracy = 0
    best_model_path = "best_modeleff000505crop.pth"
    start_time = time.time()  # Record start time

    for epoch in range(n_epochs):
        print("Epoch number %d " % (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (running_correct / total) * 100
        # log current time
        pause = time.time()
        print("Time elapsed: {:.2f} seconds".format(pause - start_time))
        print("Training dataset - Classified %d out of %d images correctly (%.3f%%). Epoch loss: %.3f" %
              (running_correct, total, epoch_accuracy, epoch_loss))

        val_accuracy = validate_model(model, val_loader)
        # Update learning rate scheduler
        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            print("Validation accuracy improved from %.3f%% to %.3f%%. Saving the model." % (
                best_val_accuracy, val_accuracy))
            best_val_accuracy = val_accuracy
            state = {'model': model.state_dict(), 'optim': optimizer.state_dict()}
            torch.save(state, best_model_path)
    end_time = time.time()  # Record end time
    duration = end_time - start_time
    print("Training completed in {:.2f} seconds.".format(duration))
    evaluate_model_on_test_set(model, test_loader)


def efficientnet_b0(num_classes=102):
    return EfficientNet(1.0, 1.0, 0.2, num_classes)


# Instantiate the custom model
custom_modal = efficientnet_b0()
device = set_device()
custom_resnet = custom_modal.to(device)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(custom_resnet.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)

# Increase the number of training epochs
n_epochs = 1200

train_network(custom_resnet, train_loader, val_loader, test_loader, loss_function, optimizer, scheduler, n_epochs)
