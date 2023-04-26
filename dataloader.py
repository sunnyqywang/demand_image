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
import h5py

from setup import data_dir
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class ImageHDF5(Dataset):
    
    def __init__(self, image_dir, data_dir, demo=False, train=True, transform=None, columns=None):
        super().__init__()
        self.shuffle_unit = 4
        self.transform = transform
        self.train = train
        self.demo = demo
        
        if train is None:
            self.files = glob.glob(image_dir+"17_*.hdf5")
            self.data_info = pd.read_csv(data_dir+"data_info.csv")
            self.demo_step = 20
            
        elif train:
            self.files = glob.glob(image_dir+"train_17_*.hdf5")
            self.data_info = pd.read_csv(data_dir+"train_data_info.csv")
            self.demo_step = 18
            
        else:
            self.files = glob.glob(image_dir+"test_17_*.hdf5")
            self.data_info = pd.read_csv(data_dir+"test_data_info.csv")
            self.demo_step = 2
            
        self.files.sort(key=natural_keys)
        
        if demo:
            self.demo_df, self.columns = load_demo_v2(data_dir)
            if columns is not None:
                self.columns = columns
                
            temp = self.data_info.drop_duplicates(subset=['state','county','tract'])
            self.demo_df = pd.merge_ordered(temp, self.demo_df, how='left', left_on=['state','county','tract'], right_on=['STATEFP','COUNTYFP','TRACTCE']).fillna(0)
            
            self.demo_df = torch.tensor(self.demo_df[self.columns].to_numpy())
            
            # print(self.demo_df)

    def __len__(self):
        return len(self.data_info) // self.shuffle_unit

    def __getitem__(self, i):        
        group_i = i//(2000//self.shuffle_unit)
        f = h5py.File(self.files[group_i], 'r')['default']
        
        idx = i % (2000//self.shuffle_unit)
        data = f[idx:idx+self.shuffle_unit]
        
        if self.transform:
            data = self.transform(torch.tensor(data))
        else:
            data = torch.tensor(data)
    
        if self.demo:
            demo_i = i // (self.demo_step // self.shuffle_unit)
            demo = self.demo_df[demo_i,:]
            demo = torch.tile(demo, (self.shuffle_unit,1))
            
            return i, data, demo
        else:
            return i, data
        
class ImageDataset(Dataset):

    def __init__(self, image_dir, data_dir, train, data_version, transform=None, sampling='clustered', image_type='png', augment=None, demo=-1):
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
                data = data[data['train_test']!=0]
            else:
                data = data[data['train_test']==0]

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
            
        self.demo = demo
        if demo > 0:
            self.demo_cs, self.demo_np = load_demo_v1(data_dir, norm=demo)
#             print(self.demo_np.shape)

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
        
        if self.demo > 0:
            census_index = self.demo_cs.index(img_name[img_name.rfind('/')+1:img_name.rfind('_')])
            census_data = self.demo_np[census_index]
            return self.image_list[idx], sample, census_data
        else:
            return self.image_list[idx], sample        
        
        
        
def image_loader(image_dir, data_dir, batch_size, num_workers, image_size, data_version, sampling='clustered', recalculate_normalize=False, image_type='png', augment=None, norm=1, demo=False, return_dataset=False):
    
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
#         normalize to [-1,+1]
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

    trainset = ImageDataset(image_dir, data_dir, train=True, data_version=data_version, transform=transform, sampling=sampling, image_type=image_type, augment=augment, demo=demo)
    testset = ImageDataset(image_dir, data_dir, train=False, data_version=data_version, transform=transform, sampling=sampling, image_type=image_type, augment=augment, demo=demo)
    
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

def load_demo_v2(data_dir):
    
    # Smart Locations from EPA
    epa = pd.read_csv(data_dir+"EPA_SmartLocations_CensusTract_export.csv")
    # log_tranform
#     epa['activity_density'] = np.log(epa['activity_density'])
#     epa['auto_oriented'] = np.log(epa['auto_oriented'])

    # Pollution PM2.5
    pm = pd.read_csv(data_dir+"PM2.5_Concentrations_2016_Illinois_Average_Annual.csv")
    pm['ctfips'] = pm['ctfips'] % 1e6
    pm['ctfips'] = pm['ctfips'].astype(int)
    demo_df = pd.merge(epa, pm, left_on=['STATEFP','COUNTYFP','TRACTCE'], right_on=['statefips','countyfips','ctfips'])

    # ACS Demo
    acs = pd.read_csv(data_dir+"demo_tract.csv")
    acs['pop_density'] = acs['tot_population'] / acs['area']
#     acs['pop_density'] = np.log(acs['pop_density'])
    demo_df = pd.merge(demo_df, acs, left_on=['COUNTYFP','TRACTCE'], right_on=['COUNTYA','TRACTA'])

    # normalize
    real_columns = ['activity_density','auto_oriented','multi_modal','pedestrian_oriented','PM2.5']
    real_columns += ['pop_density','inc_per_capita']

#     return demo_df, None

    for c in real_columns:
        # demo_df[c] = (demo_df[c] - demo_df[c].mean())/demo_df[c].std()
        demo_df[c] = (demo_df[c]-demo_df[c].min())/(demo_df[c].max()-demo_df[c].min())
        demo_df[c] = (demo_df[c]-0.5)/0.5

    pct_columns = ['employment_entropy','pop_income_entropy','wrk_emp_balance']
    pct_columns += ['pct25_34yrs','pct35_50yrs','pctover65yrs',
                 'pctwhite_alone','pct_nonwhite']
    for c in pct_columns:
        demo_df[c] = (demo_df[c]-demo_df[c].min())/(demo_df[c].max()-demo_df[c].min())
        demo_df[c] = (demo_df[c] - 0.5)/0.5

#     columns = ['pop_density','inc_per_capita','pct25_34yrs','pct35_50yrs','pctover65yrs',
#                  'pctwhite_alone','pct_nonwhite']
    columns = ['activity_density','multi_modal','pedestrian_oriented','PM2.5',
               'inc_per_capita','employment_entropy','wrk_emp_balance','pct25_34yrs']
    
    return demo_df, columns

def load_demo_v1(data_dir, norm=2):

    demo_df = pd.read_csv(data_dir+"demo_tract.csv")
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

    for d in ['pop_density','pct25_34yrs','pct35_50yrs','pctover65yrs',
                 'pctwhite_alone','pct_nonwhite',
                 'pctblack_alone',
                 'pct_col_grad','avg_tt_to_work','inc_per_capita']:
        if (norm == 0) or (norm == 'minmax'):
            demo_df[d] = demo_df[d]/demo_df[d].max()
        elif (norm == 1) or (norm == 'norm'):
            demo_df[d] = (demo_df[d]-demo_df[d].mean())/demo_df[d].std()
        elif norm == 2:
            if d[:3] == 'pct':
                demo_df[d] = (demo_df[d] - 0.5)/0.5
            else:
                demo_df[d] = demo_df[d]/demo_df[d].max()
                demo_df[d] = (demo_df[d] - 0.5)/0.5
  
# 7 variables version
#     demo_np = demo_df[['pop_density','pct25_34yrs','pct35_50yrs','pctover65yrs',
#              'pctwhite_alone','pct_nonwhite','inc_per_capita']].to_numpy()

# # 10 variables version
    demo_np = demo_df[['pop_density','pct25_34yrs','pct35_50yrs','pctover65yrs',
                 'pctwhite_alone','pct_nonwhite',
                 'pctblack_alone',
                 'pct_col_grad','avg_tt_to_work','inc_per_capita']].to_numpy()
    demo_cs = demo_df['geoid'].tolist()
    
    return demo_cs, demo_np
