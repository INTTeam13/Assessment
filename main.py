from multiprocessing import freeze_support

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


class CustomCNN(nn.Module):
    def __init__(self, num_classes=102):
        super(CustomCNN, self).__init__()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Calculate the input size for the fully connected layer
        fc1_input_size = self._get_fc1_input_size()

        self.fc1 = nn.Linear(fc1_input_size, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.6)

    def _get_fc1_input_size(self):
        dummy_input = torch.zeros(1, 3, 224, 224)
        output = self._forward_features(dummy_input)
        return output.view(-1).shape[0]

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.max_pool(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.max_pool(x)

        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
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
train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', transform=data_transforms_train,
                                                download=True)
val_dataset = torchvision.datasets.Flowers102(root='./data', split='val', transform=data_transforms_test_val, download=True)
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

def train_network(model, train_loader, test_loader, loss_function, optimizer, n_epochs):
    device = set_device()

    for epoch in range(n_epochs):
        print("Epoch number %d " % (epoch+1))
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

            outputs = model(images)  # Models classification of the images

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

# Load the best model and evaluate on the test dataset
def test_saved_model(test_loader, model_path="best_model.pth"):
    # Load the saved model
    saved_model = CustomCNN()
    device = set_device()
    saved_model = saved_model.to(device)

    checkpoint = torch.load(model_path)
    saved_model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate the model on the test dataset
    evaluate_model_on_test_set(saved_model, test_loader)


# Test the saved model with the test dataset
test_saved_model(test_loader)
#
# # Instantiate the custom model
# custom_resnet = CustomResNet(ResidualBlock)
# device = set_device()
# custom_resnet = custom_resnet.to(device)  # Transfer the model to the GPU
# loss_function = nn.CrossEntropyLoss()
#
# optimizer = optim.SGD(custom_resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)
#
# train_network(custom_resnet, train_loader, test_loader, loss_function, optimizer, 400)