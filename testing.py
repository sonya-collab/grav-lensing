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
    def __init__(self):
        pass
    def load_fits(self,filename):
        hdulist=fits.open(filename, 'update')
        hdu=hdulist[0]
        image_data=hdu.data
        hdulist.close()
        return image_data

class Initialisation:
    loader=DatasetFolder()
    fits_dataset = loader.load_fits(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project")
    #new=os.startfile(r"C:\Users\sonya\gravLensing\gravData\tum_project_lens_classif.tar\tum_project\lens_1")
    #trainloader = torch.utils.data.DataLoader(fits_dataset, batch_size=16, shuffle=True, num_workers=2)
    images = enumerate(fits_dataset)
    #image_file = get_pkg_data_filename(r"lens_1.fits")
    #fits_dataset = fits.open(r"lens_1.fits")
    ##fits_dataset.info()

    image_data = fits.getdata(images, ext=0)
    
    imageRGB_reshape = np.einsum('kij->ijk',image_data)
    plt.imshow(imageRGB_reshape)
    plt.show
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
    
    