from multiprocessing import freeze_support

# torchviz

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import time
import random
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self, block, num_classes=102):
        super(CustomResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


data_transforms_test_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# loading dataset directly from torchvision as suggested in our paper and we split the dataset into train, val, and test
train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', transform=data_transforms_train,
                                                download=True)
val_dataset = torchvision.datasets.Flowers102(root='./data', split='val', transform=data_transforms_test_val, download=True)
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
def train_network(model, train_loader, n_epochs, loss_function, optimizer):
    device = set_device()
    start_time = time.time()

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
        pause = time.time()
        print("Time elapsed: {:.2f} seconds".format(pause - start_time))
        print("Training dataset - Classified %d out of %d images correctly (%.3f%%). Epoch loss: %.3f" %
              (running_correct, total, epoch_accuracy, epoch_loss))

    end_time = time.time()
    duration = end_time - start_time
    print("Training completed in {:.2f} seconds.".format(duration))

# Modify the train_network function to include validation and model saving
def main():
    device = set_device()

    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    weight_decays = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    n_epochs = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
    best_val_accuracy = 0
    best_hyperparameters = {}
    num_searches = 20  # Number of random searches

    for _ in range(num_searches):
        lr = random.choice(learning_rates)
        wd = random.choice(weight_decays)
        ne = random.choice(n_epochs)
        print(f"Training with learning rate {lr}, weight decay {wd}, and {ne} epochs")
        custom_resnet = CustomResNet(ResidualBlock)
        custom_resnet = custom_resnet.to(device)

        loss_function = nn.CrossEntropyLoss()

        optimizer = optim.SGD(custom_resnet.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

        train_network(custom_resnet, train_loader, ne, loss_function, optimizer)

        val_accuracy = validate_model(custom_resnet, val_loader)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_hyperparameters = {'lr': lr, 'wd': wd, 'ne': ne}
            print(f"New best validation accuracy: {best_val_accuracy:.3f}% with hyperparameters {best_hyperparameters}")

    print(f"Best validation accuracy: {best_val_accuracy:.3f}% with hyperparameters {best_hyperparameters}")

main()
