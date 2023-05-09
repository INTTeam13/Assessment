import torch
import torchvision.transforms as transforms
import torch.nn as nn
import math
from torch.utils.tensorboard import SummaryWriter

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


def efficientnet_b0(num_classes=102):
    return EfficientNet(1.0, 1.0, 0.2, num_classes)


def test_saved_model(model_path="best_model.pth"):
    # Load the saved model
    saved_model = efficientnet_b0()
    print("Loading saved model from: " + model_path)
    device = set_device()
    saved_model = saved_model.to(device)
    checkpoint = torch.load(model_path, map_location=device)  # Added map_location parameter

    saved_model.load_state_dict(checkpoint['model'])

    # Create a summary writer
    writer = SummaryWriter()

    # Add the graph of the model to TensorBoard
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    writer.add_graph(saved_model, dummy_input)

    # # Export the model to ONNX format
    # torch.onnx.export(saved_model, dummy_input, "efficientnet_b0.onnx", input_names=["input"], output_names=["output"],
    #                   verbose=True)
    #
    # # Launch Netron to visualize the ONNX model
    # netron.start("efficientnet_b0.onnx")


# Test the saved model with the test dataset
test_saved_model()