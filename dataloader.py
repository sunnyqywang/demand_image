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
from setup import data_dir

class ImageDataset(Dataset):

    def __init__(self, image_dir, data_dir, train, transform=None, sampling='cluster'):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        data = pd.read_csv(data_dir+"census_tracts_filtered.csv")
        self.image_list = []

        if sampling == 'clustered':
            if train:
                data = data[data['train_test']==0]
            else:
                data = data[data['train_test']==1]

            tracts = [str(s)+'_'+str(c)+'_'+str(t) for (s,c,t) in zip(data['state_fips'], data['county_fips'], data['tract_fips'])]

            for f in tracts:
                self.image_list += glob.glob(image_dir+f+"_*.png")
        
        if sampling == 'stratified':
            for f in data['geoid']:
                im = glob.glob(image_dir+f+"_*.png")
                im = sorted(im)
                n = int(len(im) * 0.1)
                if train:
                    self.image_list += im[:-n]
                else:
                    self.image_list += im[-n:]
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.image_list[idx])
        sample = cv2.imread(img_name)
        # print(sample/255)
        # print(sample.shape)   

        if self.transform:
            sample = self.transform(sample)
        return self.image_list[idx], sample
    
    
def image_loader(image_dir, data_dir, batch_size, num_workers, image_size, sampling='cluster', recalculate_normalize=False):
    
    if recalculate_normalize:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])       
        trainset = ImageDataset(image_dir, data_dir, train=False, transform=transform, sampling=sampling)
        
        all_images = trainset[0][1].reshape(3, -1)
        for i in range(1,len(trainset)):
            all_images = torch.cat((all_images, trainset[i][1].reshape(3, -1)), dim=1)
        mean = torch.mean(all_images, axis=1)
        std = torch.std(all_images, axis=1)

        print("Image Mean: ", mean)
        print("Image Std:", std)
    else:
        mean = [0.3733, 0.3991, 0.3711]
        std = [0.2173, 0.2055, 0.2143]
        
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.Normalize(mean, std)
    ])

    trainset = ImageDataset(image_dir, data_dir, train=True, transform=transform, sampling=sampling)
    testset = ImageDataset(image_dir, data_dir, train=False, transform=transform, sampling=sampling)
    
#     print(len(trainset))
#     print(len(testset))
    
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


class SurveyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return tuple((self.x[idx], self.y[idx]))

def train_test_split(x):
    
    data = pd.read_csv(data_dir+"census_tracts_filtered.csv")
    data = pd.merge(x, data, on='geoid')
    data = data.sort_values(by='geoid')
    
    return data

def load_aggregate_travel_behavior(file, unique_ct=None):
    
    df = pd.read_csv(data_dir+file)
    df['mode_share'] = df['wtperfin_mode']/df['wtperfin_all']
    df_pivot = pd.pivot_table(df, values='mode_share', 
                              index=['state_1','state_fips_1','county_fips_1','tract_fips_1'], columns=['mode'])

    trpgen = df.groupby(['state_1','state_fips_1','county_fips_1','tract_fips_1']).mean()['wtperfin_all']

    df_pivot = pd.merge(df_pivot, trpgen, on=['state_1','state_fips_1','county_fips_1','tract_fips_1'])
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={'wtperfin_all':'trpgen',1:'active',2:'auto',3:'mas',4:'pt'}, inplace=True)
    df_pivot['geoid'] = df_pivot['state_fips_1'].astype(str)+"_"+df_pivot['county_fips_1'].astype(str)+"_"+df_pivot['tract_fips_1'].astype(str)
    if unique_ct is not None:
        df_pivot = df_pivot[df_pivot['geoid'].isin(unique_ct)]
    df_pivot.sort_values(by='geoid',inplace=True)
    # turn trip generation units to 1k trips
    df_pivot['trpgen'] = df_pivot['trpgen']/1000
    
    data = train_test_split(df_pivot)
    
    return data


def load_demo(data_dir, norm='minmax'):

    demo_df = pd.read_csv(data_dir+"demo_tract.csv")
    demo_df['census_tract'] = '17_'+demo_df['COUNTYA'].astype(str)+'_'+demo_df['TRACTA'].astype(str)

    for d in ['tot_population','pct25_34yrs','pct35_50yrs','pctover65yrs',
             'pctwhite_alone','pct_nonwhite','pctblack_alone',
             'pct_col_grad','avg_tt_to_work','inc_per_capita']:
        if norm == 'minmax':
            demo_df[d] = demo_df[d]/demo_df[d].max()
        elif norm == 'standard':
            demo_df[d] = (demo_df[d]-demo_df[d].mean())/demo_df[d].std()
            
    demo_np = demo_df[['tot_population','pct25_34yrs','pct35_50yrs','pctover65yrs',
             'pctwhite_alone','pct_nonwhite','pctblack_alone',
             'pct_col_grad','avg_tt_to_work','inc_per_capita']].to_numpy()
    demo_cs = demo_df['census_tract'].tolist()
    
    return demo_cs, demo_np