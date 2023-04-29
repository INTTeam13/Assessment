from multiprocessing import freeze_support

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)  # removed num_workers=2


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


# Set up nn model, resnet18 is a good pre-made model
resnet18_model = models.resnet18(weights=None)
num_features = resnet18_model.fc.in_features
number_of_classes = 102
resnet18_model.fc = nn.Linear(num_features, number_of_classes)
device = set_device()
resnet18_model = resnet18_model.to(device)
loss_function = nn.CrossEntropyLoss()  # change loss function here, try ReLU and see if any improvement

# Will implement mini-batch gradient descent as we specified the batch size, not stochastic gradient descent
# Change learning rate to find a good level between 0.01-0.001 and go from there
# weight_decay - helps reduce over-fitting
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

train_network(resnet18_model, train_loader, test_loader, loss_function, optimizer, 36)
