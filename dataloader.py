from PIL import Image
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

    def __init__(self, image_dir, data_dir, image_size, train, data_version, sampling='clustered', image_type='png', augment=None, demo=-1):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        
        image_list = glob.glob(self.image_dir + "*.png")
        image_list += glob.glob(self.image_dir + "*.jpg")
        self.image_df = pd.DataFrame(image_list, columns=['img_dir'])
        self.image_df['geoid'] = [img_name[img_name.rfind('/') + 1:img_name.rfind('_')] for img_name in image_list]
        self.image_df['idx'] = [int(img_name[img_name.rfind('_') + 1:img_name.rfind('.')]) for img_name in image_list]
        
        # filter images based on survey availability and train/test split
        data = pd.read_csv(data_dir+"TrainTestSplit/census_tracts_filtered-"+data_version+".csv")
        data['geoid'] = [str(s)+'_'+str(c)+'_'+str(t) for (s,c,t) in zip(data['state_fips'], data['county_fips'], data['tract_fips'])]
            
        if sampling == 'clustered':

            if train:
                include_tract = data[data['train_test']!=0]['geoid'].tolist()
            else:
                include_tract = data[data['train_test']==0]['geoid'].tolist()

            self.image_df = self.image_df[self.image_df['geoid'].isin(include_tract)]
            
        if sampling == 'stratified':
            include_tract = data['geoid'].tolist()
        
            self.image_df = self.image_df[self.image_df['geoid'].isin(include_tract)]
            
            train_split_index = int(self.image_df['idx'].max()*0.9)
            if train:
                self.image_df = self.image_df[self.image_df['idx'] < train_split_index]
            else:
                self.image_df = self.image_df[self.image_df['idx'] >= train_split_index]
        
        self.image_list = self.image_df['img_dir'].to_numpy()
                
        print(len(self.image_list), "images in dataset")
        
        self.num_unique = len(self.image_list)
        
        if augment:
            self.image_list = self.image_list + self.image_list + self.image_list
            
        self.demo = demo
        if demo >= 0:
            self.demo_cs, self.demo_np = load_demo_v1(data_dir, norm=demo)
#             print(self.demo_np.shape)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_list[idx]
        image = Image.open(img_name)

        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        image = np.array(image).astype(np.uint8)
        image = (image.astype(np.float32) / 127.5) - 1.0
        s = int((600 - self.image_size) // 2)
        e = int((600 + self.image_size) // 2)
        image = image[s:e, s:e, :]
        
        if self.augment:
            raise NotImplementedError()
#             if idx>self.num_unique*2:
#                 rotate = torchvision.transforms.RandomRotation(25)
#                 sample = rotate(sample)
#             elif idx > self.num_unique:
#                 hflip = torchvision.transforms.RandomHorizontalFlip(1)
#                 sample = hflip(sample)
        
        if self.demo >= 0:
            census_index = self.demo_cs.index(img_name[img_name.rfind('/')+1:img_name.rfind('_')])
            census_data = self.demo_np[census_index]
            return img_name, image, census_data
        else:
            return img_name, image        
        
        
        
def image_loader(image_dir, data_dir, batch_size, num_workers, image_size, data_version, sampling='clustered', augment=None, demo=-1, return_dataset=False):
    
    trainset = ImageDataset(image_dir, data_dir, image_size=image_size, train=True, data_version=data_version, sampling=sampling, augment=augment, demo=demo)
    testset = ImageDataset(image_dir, data_dir, image_size=image_size, train=False, data_version=data_version, sampling=sampling, augment=augment, demo=demo)
    
    if not return_dataset:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

        return train_loader, test_loader
    else:
        return trainset, testset
    
class SurveyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return tuple((self.x[idx], self.y[idx]))

def train_test_split_data(x, data_version):
    
    data = pd.read_csv(data_dir+"TrainTestSplit/census_tracts_filtered-"+data_version+".csv")
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
    
    census_area = pd.read_csv(data_dir+"Census_old/demo_tract.csv")[['geoid','area']]
    df_pivot = df_pivot.merge(census_area, on='geoid')
    data = train_test_split_data(df_pivot, data_version)
    
    return data

def load_demo_v2(data_dir, norm=2):
    
    demo_df = pd.read_csv(data_dir+"EPA_SmartLocations_CensusTract.csv")
    
    real_columns = ['AutoOwn0','AutoOwn1','AutoOwn2p','Workers','R_LowWageWk','R_MedWageWk','R_HiWageWk',

#     ['TotPop','CountHU','HH',
#  'TotEmp','E5_Ret','E5_Off','E5_Ind','E5_Svc','E5_Ent',
                     'E_LowWageWk','E_MedWageWk','E_HiWageWk',
#  'roads','int_mm3','int_mm4','int_po3','int_po4','emp_transit_025','emp_transit_050',
#  'D5AR','D5AE','D5BR','D5BE',
 'NatWalkInd',
 'D1a','D1b','D1c','D1c5_Ret','D1c5_Off','D1c5_Ind','D1c5_Svc','D1c5_Ent','D1d',
 'D2a_JpHH','D3a','D3b','D4b025','D4b050']
    
    pct_columns = ['D5cr','D5ce','D5dr','D5de','D5cri','D5cei','D5dri','D5dei']

    for c in real_columns:
        demo_df[c] = (demo_df[c] - demo_df[c].mean())/demo_df[c].std()
    for c in pct_columns:
        demo_df[c] = (demo_df[c] - 0.5)/0.5

    demo_cs = demo_df['geoid'].tolist()
    
    demo_np = demo_df[['AutoOwn0','AutoOwn2p','R_LowWageWk','R_HiWageWk','E_LowWageWk','E_HiWageWk','D1a','D1b','D1c',
                      'D2a_JpHH','D4b050','NatWalkInd']].to_numpy()
    
    return demo_cs, demo_np

def load_demo_v1(data_dir, norm=2):

    demo_df = pd.read_csv(data_dir+"Census_old/demo_tract.csv")
    demo_df['pop_density'] = demo_df['tot_population'] / demo_df['area']

    if norm == 3:
        high_inc = np.percentile(demo_df['inc_per_capita'].to_numpy(), 75)
        high_dens = np.percentile(demo_df['pop_density'].to_numpy(), 75)
        senior = np.percentile(demo_df['pctover65yrs'].to_numpy(), 75)
        youngad = np.percentile(demo_df['pct25_34yrs'].to_numpy(), 75)
        high_edu = np.percentile(demo_df['pct_col_grad'].to_numpy(), 75)
        
        demo_df['high_inc'] = (demo_df['inc_per_capita'] > high_inc).astype(np.float64)
        demo_df['high_dens'] = (demo_df['pop_density'] > high_dens).astype(np.float64)
        demo_df['senior'] = (demo_df['pctover65yrs'] > senior).astype(np.float64)
        demo_df['youngad'] = (demo_df['pct25_34yrs'] > youngad).astype(np.float64)
        demo_df['high_edu'] = (demo_df['pct_col_grad'] > high_edu).astype(np.float64)
    
#         return demo_df['geoid'].tolist(), demo_df[['high_inc', 'high_dens', 'senior', 'youngad', 'high_edu']].to_numpy()
        return demo_df['geoid'].tolist(), demo_df['high_dens'].to_numpy()

    else:
        for c in ['pop_density', 'pct25_34yrs', 'pct35_50yrs', 'pctover65yrs',
                  'pctwhite_alone', 'pct_nonwhite',
                  'pctblack_alone',
                  'pct_col_grad', 'avg_tt_to_work', 'inc_per_capita']:

            demo_df[c] = (demo_df[c] - demo_df[c].min()) / (demo_df[c].max() - demo_df[c].min())
            demo_df[c] = (demo_df[c] - 0.5) / 0.5

  
# 7 variables version
#     demo_np = demo_df[['pop_density','pct25_34yrs','pct35_50yrs','pctover65yrs',
#              'pctwhite_alone','pct_nonwhite','inc_per_capita']].to_numpy()

# 10 variables version
    demo_np = demo_df[['pop_density','pct25_34yrs','pct35_50yrs','pctover65yrs',
                 'pctwhite_alone','pct_nonwhite',
                 'pctblack_alone',
                 'pct_col_grad','avg_tt_to_work','inc_per_capita']].to_numpy()
    demo_cs = demo_df['geoid'].tolist()
    
    return demo_cs, demo_np
