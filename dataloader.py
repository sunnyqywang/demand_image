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

    def __init__(self, image_dir, data_dir, train, data_version, transform=None, sampling='clustered', image_type='png', augment=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.augment = augment
        
        data = pd.read_csv(data_dir+"census_tracts_filtered-"+data_version+".csv")
        self.image_list = []

        if sampling == 'clustered':
            if train:
                data = data[data['train_test']==0]
            else:
                data = data[data['train_test']==1]

            tracts = [str(s)+'_'+str(c)+'_'+str(t) for (s,c,t) in zip(data['state_fips'], data['county_fips'], data['tract_fips'])]

            for f in tracts:
                self.image_list += glob.glob(image_dir+f+"_*."+image_type)
        
        if sampling == 'stratified':
            for f in data['geoid']:
                im = glob.glob(image_dir+f+"_*."+image_type)
                im = sorted(im)
                n = int(len(im) * 0.1)
                if train:
                    self.image_list += im[:-n]
                else:
                    self.image_list += im[-n:]
                    
        print(len(self.image_list), "images in dataset")
        
        self.num_unique = len(self.image_list)
        
        if augment:
            self.image_list = self.image_list + self.image_list + self.image_list
            
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.image_list[idx])
        sample = cv2.imread(img_name)

        if self.transform:
            sample = self.transform(sample)
            
        if self.augment:
            if idx>self.num_unique*2:
                rotate = torchvision.transforms.RandomRotation(25)
                sample = rotate(sample)
            elif idx > self.num_unique:
                hflip = torchvision.transforms.RandomHorizontalFlip(1)
                sample = hflip(sample)
                
        return self.image_list[idx], sample

def image_loader(image_dir, data_dir, batch_size, num_workers, image_size, data_version, sampling='clustered', recalculate_normalize=False, image_type='png', augment=None, norm=1):
    
    if recalculate_normalize:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])       
        trainset = ImageDataset(image_dir, data_dir, train=False, transform=transform, sampling=sampling, data_version=data_version)
        
        all_images = trainset[0][1].reshape(3, -1)
        for i in range(1,len(trainset)):
            all_images = torch.cat((all_images, trainset[i][1].reshape(3, -1)), dim=1)
        mean = torch.mean(all_images, axis=1)
        std = torch.std(all_images, axis=1)

        print("Image Mean: ", mean)
        print("Image Std:", std)
    else:
        if 'zoom13' in image_dir:
            mean = [0.3733, 0.3991, 0.3711]
            std = [0.2173, 0.2055, 0.2143]
        else:
            mean = [0.3816, 0.4169, 0.3868]
            std = [0.1960, 0.1848, 0.2052]
            
            
    if norm == 1:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.Normalize(mean, std),
#             torchvision.transforms.RandomHorizontalFlip(),
#             torchvision.transforms.CenterCrop(224),
#             torchvision.transforms.Resize(image_size),
    #         torchvision.transforms.ToTensor()
        ])
    elif norm == 0:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
#             torchvision.transforms.CenterCrop(image_size),
#             torchvision.transforms.Normalize(mean, std),
#             torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Resize(image_size),
    #         torchvision.transforms.ToTensor()
        ])
    elif norm == 2:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
#             torchvision.transforms.RandomHorizontalFlip(),
#             torchvision.transforms.CenterCrop(224),
#             torchvision.transforms.Resize(image_size),
    #         torchvision.transforms.ToTensor()
        ])

    trainset = ImageDataset(image_dir, data_dir, train=True, data_version=data_version, transform=transform, sampling=sampling, image_type=image_type, augment=augment)
    testset = ImageDataset(image_dir, data_dir, train=False, data_version=data_version, transform=transform, sampling=sampling, image_type=image_type, augment=augment)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    
    return train_loader, test_loader



class ImageDataset_DR(Dataset):

    def __init__(self, image_dir, data_dir, train, target, data_version, individual=False, transform=None, num_images_per_tract=10):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.num_images_per_tract = num_images_per_tract
        self.individual = individual
        
        data = pd.read_csv(data_dir+"census_tracts_filtered-"+data_version+".csv")
        # target variables y
        file = "origin_trip_behavior.csv"
        df_pivot = load_aggregate_travel_behavior(file)

        train_test_index = df_pivot['train_test'].astype(bool)
        yy = torch.tensor(df_pivot[target].to_numpy(), dtype=torch.float)
        
        if train:
            data = data[data['train_test']==0]
            self.y = yy[~train_test_index]
        else:
            data = data[data['train_test']==1]
            self.y = yy[train_test_index]
              
        self.tracts = [str(s)+'_'+str(c)+'_'+str(t) for (s,c,t) in zip(data['state_fips'], data['county_fips'], data['tract_fips'])]
            
    def __len__(self):
        return len(self.tracts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ret_list = []
        for i in range(self.num_images_per_tract):
            img_name = os.path.join(self.image_dir, self.tracts[idx]+"_"+str(i)+".png")
            sample = cv2.imread(img_name)

            if self.transform:
                sample = self.transform(sample)
            ret_list.append(sample)
            
        return torch.stack(ret_list), self.y[idx]

def image_loader_DR(image_dir, data_dir, batch_size, num_workers, image_size, target, data_version, num_images_per_tract=10, recalculate_normalize=False):
    
    if recalculate_normalize:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])       
        trainset = ImageDataset(image_dir, data_dir, train=False, transform=transform, sampling=sampling, data_version=data_version)
        
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

    trainset = ImageDataset_DR(image_dir, data_dir, data_version=data_version, train=True, target=target, transform=transform, num_images_per_tract=num_images_per_tract)
    testset = ImageDataset_DR(image_dir, data_dir, data_version=data_version, train=False, target=target, transform=transform, num_images_per_tract=num_images_per_tract)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    
    return train_loader, test_loader


class SurveyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return tuple((self.x[idx], self.y[idx]))

def train_test_split_data(x, data_version):
    
    data = pd.read_csv(data_dir+"census_tracts_filtered-"+data_version+".csv")
    data = pd.merge(x, data, on='geoid')
    data = data.sort_values(by='geoid')
    
    return data

def load_aggregate_travel_behavior(file, data_version):
    
    df = pd.read_csv(data_dir+file)
    df['mode_share'] = df['wtperfin_mode']/df['wtperfin_all']
    df_pivot = pd.pivot_table(df, values='mode_share', 
                              index=['state_fips_1','county_fips_1','tract_fips_1'], columns=['mode'])

    trpgen = df.groupby(['state_fips_1','county_fips_1','tract_fips_1']).mean()['wtperfin_all']
    df_pivot = pd.merge(df_pivot, trpgen, on=['state_fips_1','county_fips_1','tract_fips_1'])
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={'wtperfin_all':'trpgen',1:'active',2:'auto',3:'mas',4:'pt'}, inplace=True)
    df_pivot['geoid'] = df_pivot['state_fips_1'].astype(str)+"_"+df_pivot['county_fips_1'].astype(str)+"_"+df_pivot['tract_fips_1'].astype(str)
    df_pivot.sort_values(by='geoid',inplace=True)
    # turn trip generation units to 1k trips`
    df_pivot['trpgen'] = df_pivot['trpgen']/1000
    
    census_area = pd.read_csv(data_dir+"demo_tract.csv")[['geoid','area']]
    df_pivot = df_pivot.merge(census_area, on='geoid')
    data = train_test_split_data(df_pivot, data_version)
    
    return data


def load_demo(data_dir, norm='minmax'):

    demo_df = pd.read_csv(data_dir+"demo_tract.csv")

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
    demo_cs = demo_df['geoid'].tolist()
    
    return demo_cs, demo_np
