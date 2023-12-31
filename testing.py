from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

import torch
from astropy.io import fits
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms as transform
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits
from PIL import Image

import pandas as pd

'''class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
    
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
    )'''
'''class FitsFolder(datasets.DatasetFolder):

    EXTENSIONS = ['.fits']

    def __init__(self, root, transform=None, target_transform=None,
                 loader=None):
        if loader is None:
            loader = self.__fits_loader

        super(FitsFolder, self).__init__(root, loader, self.EXTENSIONS,
                                         transform=transform,
                                         target_transform=target_transform)

    @staticmethod
    def __fits_loader(filename):
        data = fits.getdata(filename)
        #return data
        return Image.fromarray(data)'''

class DatasetFolder:
    def __init__(self, root_dir, transforms=None):
        self.root_dir=root_dir
        self.transforms=transforms
    def __len__(self):
        return len(os.listdir(self.root_dir))
    def __getitem__(self,index):
        label=os.listdir(self.root_dir)[index]
        fits_path=r''+self.root_dir+'/'+label
        file=fits.getdata(fits_path, ext=0)
        file=file.astype(float)
        #print(type(file), file.dtype)
        #hdulist=fits.open(filename, 'update')
        #hdu=hdulist[0]
        #image_data=hdu.data
        #hdulist.close()
        if self.transforms:
            file=self.transforms(file)
        return (file, label)

####testing for image channels, normalisations etc. non-working alternatives:
#transforms.Normalize((0,0,0),(1,1,1)) ,transforms.Normalize((0,),(1,)) , transforms.Normalize(mean=(0.0628418,  0.30500231, 0.67452615),std=(0.16051376, 0.73852639, 1.74141281)) wrong!!! has to be done for each! picture
        #doesn't work yet!
        #Assume your image is a 4D tensor of shape (N, H, W, C), want to move C to front

#print(len(lenses[1][100][0][0][3]))

#print(type(train_dataset[0][0]))
        #for traindata in train_dataset:
        #   train_reshape = np.einsum('kij->ijk',traindata)

#print(type(train_dataset[0]))

'''mean=[]
        std=[]
        for i in range(3):
            meantemp, stdtemp = target[i].mean([0,0,0]), target.std([1,1,1])
            mean.append(meantemp)
            std.append(stdtemp)
        print('mean:', mean, 'std:', std)'''

# Calculate the mean and standard deviation
        #mean /= len(trainloader.dataset)
        #std /= len(trainloader.dataset)
        #first50=train_dataset[:50]
        #target=next(iter(trainloader))
        #first_50_indices = range(50)
        #subset_dataset = torch.utils.data.Subset(target, first_50_indices)
        #std = torch.std(torch.stack([subset_dataset[i][0] for i in range(len(subset_dataset))]), dim = 1)
        #mean = torch.mean(torch.stack(first50), dim=(0, 2, 3))

        #npim=np.array(subset_dataset)
        #mean, std = np.mean(npim, axis=0), np.std(npim, axis=0)
        #print('mean:', mean, 'std:', std)

#alternative images grid
#print(len(target[0]))
        #img_grid = torchvision.utils.make_grid(target[0]) #should show images in grid with oorresponding labels...
        #print(len(img_grid[0]))
        #imgGrid_reshape = np.einsum('ijk->kij',img_grid)
        #print(len(imgGrid_reshape))
        #plt.imshow(img_grid)
        #, one_channel=True
class Initialisation:
    def main():
        filedir=r'C:/Users/sonya/gravLensing/gravlensing_sonya.francisco/tum_project/lens'
        filename1=os.listdir(filedir)
        print(filename1)
        lenses = []
        for filename in filename1:
            filedir2=r''+filedir+'/'+filename
            filename2=os.listdir(filedir2)
            data=DatasetFolder(filedir2, None)
            lenses.append(data)
        print(lenses)
        #lenses.append(loader.load_fits(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project\lens\lens_2"))
        #lenses.append(loader.load_fits(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project\lens\lens_3"))
        #lenses.append(loader.load_fits(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project\lens\lens_4"))
        #lenses.append(loader.load_fits(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project\lens\lens_5"))
        #l.append(datasets.ImageFolder(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project\lens"))
        #l.append(datasets.ImageFolder(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project\lens\lens_4"))
        #l.append(datasets.ImageFolder(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project\lens\lens_5"))
        image_dataset = torch.utils.data.ConcatDataset(lenses)
        #print(image_dataset)
        trainloader = torch.utils.data.DataLoader(image_dataset, batch_size=64, shuffle=True)
        

        
        
        #new=os.startfile(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project\lens_1")
        #trainloader = torch.utils.data.DataLoader(fits_dataset, batch_size=16, shuffle=True, num_workers=2)
        #print(trainloader)
       
        
        target=next(iter(trainloader))
        print(len(trainloader))
        
        fig=plt.figure()
        #plt.imshow()
        imageRGB_reshape = np.einsum('kij->ijk',target[0][1])
        plt.imshow(imageRGB_reshape)
        plt.show()
        #image_file = get_pkg_data_filename(r"lens_1.fits")
        #fits_dataset = fits.open(r"len
        # lens_1.fits")
        #fits_dataset.info()

        
        
        #
        #image=fits.getdata()
        
        #my_dataset = CustomDataSet(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project\lens_1", transform=trsfm)
        #train_loader = data.DataLoader(my_dataset , batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        #for idx, img in enumerate(train_loader):
        #do your training now
        #transform=transform.Compose([transform.ToTensor(), transform.Normalize((0,0,0),(1,1,1))])
        #trainloader = torch.utils.data.DataLoader(fits_dataset, batch_size=16, shuffle=True, num_workers=2)
        #validationloader = torch.utils.data.DataLoader(fits_dataset, batch_size=16, shuffle=True, num_workers=2)
        #classes = ('lens', 'non_lens')

        

        # functions to show an image


        #def imshow(img):
            #img = img / 2 + 0.5     # unnormalize
        #   npimg = img.numpy()
        #  plt.imshow(np.transpose(npimg, (1, 2, 0)))


        # get some random training images
        #images = iter(trainloader)
        

        # show images
    if __name__=='__main__':
        main() 
    