#Hello there! This is just a test! HELLO

import torch
from astropy.io import fits
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transform
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits

class FITSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        file_name = os.listdir(self.root_dir)[idx]
        file_path = os.path.join(self.root_dir, file_name)

        with fits.open(file_path) as hdul:
            data = hdul[0].data
        if self.transform:
            data = self.transform(data)
        return data
# Instantiate the dataset


# Create a DataLoader




class Initialisation:
    '''training_data = torchvision.datasets.FashionMNIST(
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
    )'''
    #fits_dataset = fits.open(r"lens_1.fits")
    transform=transform.Compose([transform.ToTensor(), transform.Normalize((0,0,0),(1,1,1))])
    loader=FITSDataset(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project", transform)
    
    fits_dataset=loader.__getitem__(0)
    train_size = int(0.8 * len(fits_dataset))
    test_size = len(fits_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(fits_dataset, [train_size, test_size])
    #trainloader = torch.utils.data.DataLoader(fits_dataset, batch_size=16, shuffle=True, num_workers=2)
    #validationloader = torch.utils.data.DataLoader(fits_dataset, batch_size=16, shuffle=False, num_workers=2)
    classes = ('lens', 'non_lens')
    images_train = iter(fits_dataset)
    #images_valid = iter(validationloader)
    train_data = fits.getdata(images_train, ext=0)
    #valid_data = fits.getdata(images_valid, ext=0)
    imageRGB_reshape_train = np.einsum('kij->ijk',images_train)
    #imageRGB_reshape_valid = np.einsum('kij->ijk',images_valid)
    plt.imshow(imageRGB_reshape_train)
    plt.show
    

    # functions to show an image


    '''def imshow(img):
        #img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))'''


    # get some random training images
    #images = iter(train_dataset)
    

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % for j in range(4)))

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
        x = F.relu(self.conv11(x), (2, 2))
        x = F.relu(self.conv12(x), (2, 2))
        x = F.relu(self.conv21(x), (2, 2))
        x = F.relu(self.conv22(x), (2, 2))
        x = F.relu(self.conv31(x), (2, 2))
        x = F.relu(self.conv32(x), (2, 2))
        x = F.relu(self.conv41(x), (2, 2))
        x = F.relu(self.conv42(x), (2, 2))
        x = F.relu(self.conv51(x), (2, 2))
        x = F.relu(self.conv52(x), (2, 2))
        x = F.relu(self.conv61(x), (2, 2))
        x = F.relu(self.conv62(x), (2, 2))
        x = F.relu(self.conv71(x), (2, 2))
        x = F.relu(self.conv72(x), (2, 2))
        x = F.relu(self.conv81(x), (2, 2))
        x = F.relu(self.conv82(x), (2, 2))
    
        # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model=Net().to(device)
print(model)
