import torch.nn as nn

# Define the CNN Model
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.block_layers(3, 8)
        self.conv2 = self.block_layers(8, 16)
        self.fc1 = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(16 * 8 * 8, 256), nn.ReLU())
        self.fc2 = nn.Linear(256, num_classes)

    def block_layers(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
