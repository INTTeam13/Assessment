import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image


# Extract features function
def extract_features(image_tensor):
    # Convert the tensor back to a numpy array and transpose the dimensions
    image = image_tensor.numpy().transpose((1, 2, 0))
    # Convert to grayscale and handle different numbers of channels
    if image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif image.shape[2] == 1:
        gray = image.astype(np.uint8)
        hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2HSV)
    else:
        raise ValueError("Unsupported number of channels in input image")

    # Compute SIFT features
    sift = cv2.xfeatures2d.SIFT_create()
    _, sift_features = sift.detectAndCompute(gray, None)

    # Compute HOG features
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True,
                       multichannel=False)

    return hsv, sift_features, hog_features



# Custom dataset class
class Flowers102WithFeatures(torchvision.datasets.Flowers102):
    def __init__(self, *args, **kwargs):
        super(Flowers102WithFeatures, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        label = self.labels[index]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Extract features
        image_np = np.array(image).astype(np.uint8)
        hsv, sift_features, hog_features = extract_features(image_np)

        hsv = torch.tensor(hsv, dtype=torch.float32)
        sift_features = torch.tensor(sift_features, dtype=torch.float32)
        hog_features = torch.tensor(hog_features, dtype=torch.float32)

        return image, hsv, sift_features, hog_features, label


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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# loading dataset directly from torchvision as suggested in our paper and we split the dataset into train, val, and test
train_dataset = Flowers102WithFeatures(root='./data', split='train', transform=data_transforms_train,
                                       download=True)
val_dataset = Flowers102WithFeatures(root='./data', split='val', transform=data_transforms_test_val, download=True)
test_dataset = Flowers102WithFeatures(root='./data', split='test', transform=data_transforms_test_val,
                                      download=True)

# Create data loaders to load the data in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)  # removed num_workers=2


class CustomResNetWithFeatures(CustomResNet):
    def forward(self, x, hsv, sift_features, hog_features):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        # Pad or truncate the SIFT features tensor to a fixed size
        max_sift_features = 300
        sift_features_padded = torch.zeros(x.size(0), max_sift_features, dtype=torch.float32)
        for i, sift_feature in enumerate(sift_features):
            if sift_feature.size(0) <= max_sift_features:
                sift_features_padded[i, :sift_feature.size(0)] = sift_feature
            else:
                sift_features_padded[i] = sift_feature[:max_sift_features]

        # Flatten the SIFT and HOG feature tensors
        sift_features_flat = sift_features_padded.view(x.size(0), -1)
        hog_features_flat = hog_features.view(x.size(0), -1)

        # Flatten the HSV tensor and normalize it to the same range as the other features
        hsv_flat = torch.flatten(hsv, start_dim=1, end_dim=3)
        hsv_flat_normalized = hsv_flat / 255.0

        # Concatenate features
        x = torch.cat((x, hsv_flat_normalized, sift_features_flat, hog_features_flat), dim=1)

        x = self.fc(x)
        return x


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
            images, hsv, sift_features, hog_features, labels = data
            images = images.to(device)
            hsv = hsv.to(device)
            sift_features = sift_features.to(device)
            hog_features = hog_features.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images, hsv, sift_features, hog_features)  # Models classification of the images

            _, predicted = torch.max(outputs.data, 1)  # 1 specifies dimension it is reduced to

            predicted_correctly_on_epoch += (labels == predicted).sum().item()

    epoch_accuracy = (predicted_correctly_on_epoch / total) * 100  # Get accuracy as percentage

    print("Test dataset - Classified %d out of %d images correctly (%.3f%%)" %
          (predicted_correctly_on_epoch, total, epoch_accuracy))


def train_network(model, train_loader, test_loader, loss_function, optimizer, n_epochs):
    device = set_device()

    for epoch in range(n_epochs):
        print("Epoch number %d " % (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, hsv, sift_features, hog_features, labels = data
            images = images.to(device)
            hsv = hsv.to(device)
            sift_features = sift_features.to(device)
            hog_features = hog_features.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images, hsv, sift_features, hog_features)  # Models classification of the images

            _, predicted = torch.max(outputs.data, 1)  # 1 specifies dimension it is reduced to

            loss = loss_function(outputs, labels)

            loss.backward()  # Back propagate through network

            optimizer.step()  # Update weights

            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (running_correct / total) * 100  # Get accuracy as percentage

        print("Training dataset - Classified %d out of %d images correctly (%.3f%%). Epoch loss: %.3f" %
              (running_correct, total, epoch_accuracy, epoch_loss))

    evaluate_model_on_test_set(model, test_loader)

    print("Finished")
    return model


# Instantiate the custom model
custom_resnet = CustomResNetWithFeatures(ResidualBlock)
device = set_device()
custom_resnet = custom_resnet.to(device)  # Transfer the model to the GPU if available
loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(custom_resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

train_network(custom_resnet, train_loader, test_loader, loss_function, optimizer, 400)
