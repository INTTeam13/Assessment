import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

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

def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)

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


data_transforms_test_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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
            print("Predicted correctly: %d out of %d" % (predicted_correctly_on_epoch, total))

    epoch_accuracy = (predicted_correctly_on_epoch / total) * 100  # Get accuracy as percentage

    print("Test dataset - Classified %d out of %d images correctly (%.3f%%)" %
          (predicted_correctly_on_epoch, total, epoch_accuracy))

test_dataset = torchvision.datasets.Flowers102(root='./data', split='test', transform=data_transforms_test_val,
                                               download=True)

# Create data loaders to load the data in batches
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)  # removed num_workers=2
def test_saved_model(test_loader, model_path="best_model.pth"):
    # Load the saved model
    saved_model = CustomResNet(ResidualBlock)
    print("Loading saved model from: " + model_path)
    device = set_device()
    saved_model = saved_model.to(device)

    checkpoint = torch.load(model_path, map_location=device)  # Added map_location parameter
    saved_model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate the model on the test dataset
    evaluate_model_on_test_set(saved_model, test_loader)

# Test the saved model with the test dataset
test_saved_model(test_loader)
