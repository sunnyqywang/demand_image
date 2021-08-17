import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch.utils.data import Dataset, TensorDataset

import torchvision
import torchvision.models
import torchvision.transforms

import cv2
import os
import glob

class ImageDataset(Dataset):

    def __init__(self, image_dir, data_dir, train, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        data = pd.read_csv(data_dir+"census_tracts_filtered.csv")
        if train:
            data = data[data['train_test']==0]
        else:
            data = data[data['train_test']==1]
        
        tracts = [str(s)+'_'+str(c)+'_'+str(t) for (s,c,t) in zip(data['state_fips'], data['county_fips'], data['tract_fips'])]
        self.image_list = []
        
        for f in tracts:
            self.image_list += glob.glob(image_dir+f+"_*.png")
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.image_list[idx])
        sample = cv2.imread(img_name)
        # print(sample/255)

        if self.transform:
            sample = self.transform(sample)
        # print(sample)   
        return self.image_list[idx], sample
    
    
def image_loader(image_dir, data_dir, batch_size, num_workers, image_size, recalculate_normalize=False):
    
    if recalculate_normalize:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])       
        trainset = ImageDataset(image_dir, data_dir, train=False, transform=transform)
        all_images = trainset[0].reshape(3, -1)
        for i in range(1,len(trainset)):
            all_images = torch.cat((all_images, trainset[i].reshape(3, -1)), dim=1)
        mean = torch.mean(all_images, axis=1)
        std = torch.std(all_images, axis=1)

        print("Satellite Mean: ", mean)
        print("Satellite Std:", std)
    else:
        mean = [0.3733, 0.3991, 0.3711]
        std = [0.2173, 0.2055, 0.2143]
        
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.Normalize(mean, std)
    ])


    trainset = ImageDataset(image_dir, data_dir, train=True, transform=transform)
    testset = ImageDataset(image_dir, data_dir, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    
    return train_loader, test_loader

def get_loader(batch_size, num_workers): # Load CIFAR data
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])

    dataset_dir = '~/.torchvision/datasets/CIFAR10'
    train_dataset = torchvision.datasets.CIFAR10(
     dataset_dir, train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
     dataset_dir, train=False, transform=test_transform, download=True)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader
