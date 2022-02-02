import sys
sys.path.append("models/")

import itertools
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from setup import *
from dataloader import SurveyDataset
import mnl


if __name__ == "__main__":
    
    args = parse_args()
    n_alts = 4
    variable_names = ['active','auto','mas','pt', 'trpgen']

    demo_variables = ['tot_population','pct25_34yrs','pct35_50yrs','pctover65yrs',
         'pctwhite_alone','pct_nonwhite','pctblack_alone',
         'pct_col_grad','avg_tt_to_work','inc_per_capita']
    
    with open(proj_dir+"latent_space/"+args.model_type+"_"+args.zoomlevel+"_"+str(args.output_dim**2*2048)+"_"+
                       args.model_run_date+".pkl", "rb") as f: 
        encoder_output = pkl.load(f)
        im = pkl.load(f)
        ct = pkl.load(f)
    
    # Aggregate Embeddings
    unique_ct = list(set(ct))
    unique_ct.sort()
    ct = np.array(ct)
    aggregate_embeddings = []
    for i in unique_ct:
        aggregate_embeddings.append(np.mean(encoder_output[ct == i], axis=0))
    aggregate_embeddings = np.array(aggregate_embeddings)
    
    # Trip Data
    tp = pd.read_csv(data_dir+"trips.csv")
    tp['tract_1'] = tp['state_fips_1'].astype(str) + '_' + tp['county_fips_1'].astype(str)+ '_' + tp['tract_fips_1'].astype(str)
    tp['tract_2'] = tp['state_fips_2'].astype(str) + '_' + tp['county_fips_2'].astype(str)+ '_' + tp['tract_fips_2'].astype(str)

    tp['morning'] = (tp['dep_hour'] > 6) & (tp['dep_hour'] < 10)
    tp['afternoon'] = (tp['dep_hour'] > 15) & (tp['dep_hour'] < 19)
    tp['morning'] = tp['morning'].astype(int)
    tp['afternoon'] = tp['afternoon'].astype(int)

    tp['const'] = 1

    def normalize_features(df, cols):
        for c in cols:
            df[c] = df[c]/df[c].max()
        return df
    
    # Filter Trips
    
    unique_ct = np.array(unique_ct)

    x_embed = []
    trip_filter = []
    for t1, t2 in zip(tp['tract_1'], tp['tract_2']):
        if sum(unique_ct == t1) == 1 and sum(unique_ct == t2) == 1:
            x_embed.append(np.hstack((aggregate_embeddings[unique_ct == t1], aggregate_embeddings[unique_ct == t2])).flatten())
            trip_filter.append(True)
        else:
            trip_filter.append(False)

    trip_filter = np.array(trip_filter)
    x_embed = np.array(x_embed)
    x_trip = tp[['morning','afternoon','companion', 'distance', 
             'from_home', 'to_home', 'purp_work', 'purp_school', 'purp_errand', 'purp_recreation', 
             'ontime_important', '12_18yrs', '18_25yrs', '25_55yrs', '55+yrs', 'no_age', 
             'disability', 'educ_col', 'educ_grad', 
             'race_white', 'race_black', 'race_asian', 
             'male', 'female', 
             'emply_park', 'emply_transit', 'emply_veh', 'emply_wfh', 'emply_flex', 'emply_hours', 
             'license', 'person_trips', 'person_transit', 'person_freq_transit', 
             'hh_inc_0_30', 'hh_inc_30_60', 'hh_inc_60_100', 'hh_inc_100_150', 'hh_inc_150', 
             'avg_pr_veh', 'home_own', 'home_house', 'home_condo']].to_numpy()[trip_filter]

    x = np.concatenate([x_trip, x_embed], axis=1)

    y = tp['mode'].astype(int).to_numpy() - 1
    y = y[trip_filter]
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    
    trainset = SurveyDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True)

    testset = SurveyDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    testloader = DataLoader(testset, batch_size=len(testset), shuffle=False)
    
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    wd_list = [0.00005,0.0001,0.0005,0.001,0.005,0.01]
    lr_list = [0.005, 0.01, 0.02]
    do_list = [0, 0.1, 0.2, 0.5]

    for (lr, wd, do) in itertools.product(lr_list, wd_list, do_list):

        model = mnl.MNL2(n_alts=n_alts, dim_embed=x_embed.shape[-1], dim_demo=x_trip.shape[-1], dropout=do)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        ref1 = 0
        ref2 = 0

        for epoch in range(100):
            loss_ = 0
            correct = 0
            for batch, (x_batch, y_batch) in enumerate(trainloader):
                # Compute prediction and loss
                util = model(x_batch)
                loss = loss_fn(util, y_batch)
                loss_ += loss.item() * len(x_batch)

                pred = torch.argmax(util, dim=1)
                correct += torch.sum(pred == y_batch)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if epoch % 3 == 0:
                loss_ /= len(trainset)
                train_acc = correct/len(trainset)
                print(f"[epoch: {epoch:>3d}] Train loss: {loss_:.4f} accuracy: {train_acc:.3f}")

                correct = 0
                test_loss_ = 0
                for batch, (x_batch, y_batch) in enumerate(testloader):
                    util = model(x_batch)
                    loss = loss_fn(util, y_batch)
                    test_loss_ += loss.item()
                    pred = torch.argmax(util, dim=1)
                    correct += torch.sum(pred == y_batch)
                assert batch == 0 # there is only one batch in test
                test_acc = correct/len(testset)            
                print(f"[epoch: {epoch:>3d}] Test loss: {test_loss_:.4f} accuracy: {test_acc:.3f}")

                if epoch > 15:
                    if (np.abs(loss_ - ref1)/ref1<ref1*0.01) & (np.abs(loss_ - ref2)/ref2<ref2*0.01):
                        print("Early stopping at epoch", epoch)
                        break
                    if (ref1 < loss_) & (ref1 < ref2):
                        print("Diverging. stop.")
                        break
                    if loss_ < best:
                        best = loss_
                        best_test = test_loss_
                        best_epoch = epoch
                        best_train_acc = train_acc
                        best_test_acc = test_acc
                else:
                    best = loss_
                    best_test = test_loss_
                    best_epoch = epoch
                    best_train_acc = train_acc
                    best_test_acc = test_acc

                ref2 = ref1
                ref1 = loss_

        with open(out_dir+args.normalization[0]+'_'+args.model_code+".csv", "a") as f:
            f.write("%s,%s,%s,%s,%.4f,%.5f,%.1f,%d,%.4f,%.4f,%.4f,%.4f\n" % \
                (args.model_run_date, args.model_type, args.zoomlevel, "MNL2", lr, wd, do, 
                 best_epoch, best, best_test, best_train_acc, best_test_acc))
