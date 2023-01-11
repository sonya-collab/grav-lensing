#Hello there! This is just a test! HELLO
import torch
from astropy.io import fits
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt



class initialisation:
    training_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform.ToTensor()
    )

    test_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform.ToTensor()
    )
    train_dataloader =torch.utils.data.DataLoader(training_data, batch_size=4, shuffle=True, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 3x3 square convolution
        # kernel
        self.conv11 = nn.Conv2d(3, 6, 3)
        self.conv12 = nn.Conv2d(6, 16, 3)

        self.conv21 = nn.Conv2d(1, 6, 3)
        self.conv22 = nn.Conv2d(6, 16, 3)

        self.conv31 = nn.Conv2d(1, 6, 3)
        self.conv32 = nn.Conv2d(6, 16, 3)

        self.conv41 = nn.Conv2d(16, 16, 3)
        self.conv42 = nn.Conv2d(16, 16, 3)

        self.conv51 = nn.Conv2d(16, 16, 3)
        self.conv52 = nn.Conv2d(16, 16, 3)

        self.conv61 = nn.Conv2d(16, 16, 3)
        self.conv62 = nn.Conv2d(16, 16, 3)

        self.conv71 = nn.Conv2d(16, 16, 3)
        self.conv72 = nn.Conv2d(16, 16, 3)

        self.conv81 = nn.Conv2d(16, 16, 3)
        self.conv82 = nn.Conv2d(16, 16, 3)

        self.fc1 = nn.Linear(16 * 72 * 72, 16)  # 6*6 from image dimension
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        for block in range(7):
            x = F.relu(self.conv11(x)), (2, 2))
            # If the size is a square you can only specify a single number
            x = F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
