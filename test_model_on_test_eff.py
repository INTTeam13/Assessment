
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from collections import defaultdict
import math

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



def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)


data_transforms_test_val = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    # Initialize a dictionary to track the number of incorrect predictions per class
    incorrect_predictions = defaultdict(int)

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            print("accuracy of this batch in percentage: ", (predicted == labels).sum().item() / labels.size(0) * 100)

            # Get indices of the incorrect predictions
            incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in incorrect_indices:
                true_label = labels[idx].item()
                incorrect_predictions[true_label] += 1

            predicted_correctly_on_epoch += (labels == predicted).sum().item()
            print("Predicted correctly: %d out of %d" % (predicted_correctly_on_epoch, total))
            print("Total Accuracy: %.3f%%" % ((predicted_correctly_on_epoch / total) * 100))

    epoch_accuracy = (predicted_correctly_on_epoch / total) * 100

    print("Test dataset - Classified %d out of %d images correctly (%.3f%%)" %
          (predicted_correctly_on_epoch, total, epoch_accuracy))

    # Print the incorrect predictions statistics
    print("Incorrect predictions statistics:")
    for class_id, count in incorrect_predictions.items():
        print(f"Class {class_id}: {count} incorrect predictions")
test_dataset = torchvision.datasets.Flowers102(root='./data', split='test', transform=data_transforms_test_val,
                                               download=True)

# Create data loaders to load the data in batches
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)  # removed num_workers=2

def efficientnet_b0(num_classes=102):
    return EfficientNet(1.0, 1.0, 0.2, num_classes)
def test_saved_model(test_loader, model_path="best_model_eff_with.pth"):
    # Load the saved model
    saved_model = efficientnet_b0()
    print("Loading saved model from: " + model_path)
    device = set_device()
    saved_model = saved_model.to(device)

    checkpoint = torch.load(model_path, map_location=device)  # Added map_location parameter
    saved_model.load_state_dict(checkpoint['model'])

    # Evaluate the model on the test dataset
    evaluate_model_on_test_set(saved_model, test_loader)

# Test the saved model with the test dataset
test_saved_model(test_loader)