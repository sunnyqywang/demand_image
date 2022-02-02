import sys
sys.path.append("models/")

from collections import OrderedDict
import pandas as pd
import pickle as pkl
import numpy as np

import itertools
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

from dataloader import SurveyDataset, load_aggregate_travel_behavior, load_demo
from M1_util_train_test import load_model, test
import mnl
from setup import out_dir, data_dir, image_dir, model_dir, proj_dir, parse_args


if __name__ == "__main__":
    
    args = parse_args()
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
    
    file = "origin_trip_behavior.csv"
    df_pivot = load_aggregate_travel_behavior(file, unique_ct)

    train_test_index = df_pivot['train_test'].astype(bool).to_numpy()
    # train_test_index = np.random.rand(len(df_pivot)) < 0.2

    y = df_pivot[variable_names].to_numpy()
    y_train = y[~train_test_index,:4]
    y_test = y[train_test_index,:4]

    x_train = aggregate_embeddings[~train_test_index, :]
    x_test = aggregate_embeddings[train_test_index, :]
    
    
    # dataloader and model definition

    trainset = SurveyDataset(torch.tensor(x_train,  dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True)

    testset = SurveyDataset(torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))
    testloader = DataLoader(testset, batch_size=len(testset), shuffle=True)

    kldivloss = nn.KLDivLoss(reduction='sum')
    mseloss = nn.MSELoss(reduction='none')


    for (lr, wd) in itertools.product(args.lr_list, args.wd_list):
        # model setup
        model = mnl.MNL(n_alts=4, n_features=x_train.shape[-1])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # model training

        ref1 = 0
        ref2 = 0

        for epoch in range(400):

            kl_ = 0
            mse_ = 0
            mse1_ = 0
            mse2_ = 0
            mse3_ = 0
            mse4_ = 0

            for batch, (x_batch, y_batch) in enumerate(trainloader):
                # Compute prediction and loss
                util = model(x_batch)
                probs = torch.log(nn.functional.softmax(util, dim=1))
                kl = kldivloss(probs, y_batch)
        #         kl = kldivloss(torch.log(util), y_batch)
                kl_ += kl.item()

                mse = mseloss(torch.exp(probs), y_batch)
        #         mse = mseloss(util, y_batch)
                mse_ += mse.sum().item()
                mse1_ += mse[:,0].sum().item()
                mse2_ += mse[:,1].sum().item()
                mse3_ += mse[:,2].sum().item()
                mse4_ += mse[:,3].sum().item()
                mse = mse.sum()

                # Backpropagation
                optimizer.zero_grad()
                kl.backward()
                optimizer.step()

            train_kl = kl_/len(trainset)
            train_mse = np.sqrt(mse_/len(trainset))
            train_mse1 = np.sqrt(mse1_/len(trainset))
            train_mse2 = np.sqrt(mse2_/len(trainset))
            train_mse3 = np.sqrt(mse3_/len(trainset))
            train_mse4 = np.sqrt(mse4_/len(trainset))

            if epoch % 10 == 0:
                print(f"[epoch: {epoch:>2d}] Train KL loss: {train_kl:.3f} \
                    RMSE {train_mse:.3f} \
                    {train_mse1:.3f} {train_mse2:.3f} {train_mse3:.3f} {train_mse4:.3f}")
            loss_ = train_kl

            if epoch % 5 == 0:
                if epoch > 50:
                    if (np.abs(loss_ - ref1)/ref1<ref1*0.01) & (np.abs(loss_ - ref2)/ref2<ref2*0.01):
                        print("Early stopping at epoch", epoch)
                        break
                    if (ref1 < loss_) & (ref1 < ref2):
                        print("Diverging. stop.")
                        break
                    if loss_ < best:
                        best = loss_
                        best_epoch = epoch
                else:
                    best = loss_
                    best_epoch = epoch

                ref2 = ref1
                ref1 = loss_

            if epoch % 10 == 0:

                kl_ = 0
                mse_ = 0 
                mse1_ = 0
                mse2_ = 0
                mse3_ = 0
                mse4_ = 0

                for batch, (x_batch, y_batch) in enumerate(testloader):
                    util = model(x_batch)
                    probs = torch.log(nn.functional.softmax(util,dim=1))
                    kl = kldivloss(probs, y_batch)
            #         kl = kldivloss(torch.log(util), y_batch)
                    kl_ += kl.item()

                    mse = mseloss(torch.exp(probs), y_batch)
            #         mse = mseloss(util, y_batch)
                    mse_ += mse.sum().item()
                    mse1_ += mse[:,0].sum().item()
                    mse2_ += mse[:,1].sum().item()
                    mse3_ += mse[:,2].sum().item()
                    mse4_ += mse[:,3].sum().item()

                test_kl = kl_/len(testset)
                test_mse = np.sqrt(mse_/len(testset))
                test_mse1 = np.sqrt(mse1_/len(testset))
                test_mse2 = np.sqrt(mse2_/len(testset))
                test_mse3 = np.sqrt(mse3_/len(testset))
                test_mse4 = np.sqrt(mse4_/len(testset))

                r1 = r2_score(y_batch.numpy()[:,0],torch.exp(probs).detach().numpy()[:,0])
                r2 = r2_score(y_batch.numpy()[:,1],torch.exp(probs).detach().numpy()[:,1])
                r3 = r2_score(y_batch.numpy()[:,2],torch.exp(probs).detach().numpy()[:,2])
                r4 = r2_score(y_batch.numpy()[:,3],torch.exp(probs).detach().numpy()[:,3])

                print(f"[epoch: {epoch:>2d}] Test KL loss: {kl_/len(testset):.3f}\
                        RMSE {np.sqrt(mse_/len(testset)):.3f} \
                        {np.sqrt(mse1_/len(testset)):.3f} {np.sqrt(mse2_/len(testset)):.3f} {np.sqrt(mse3_/len(testset)):.3f} {np.sqrt(mse4_/len(testset)):.3f}")
                print(f"\t\t\t\t\t\t\tR2 score: {r1:.3f} {r2:.3f} {r3:.3f} {r4:.3f} ")


        with open(out_dir+args.sampling[0]+"_"+args.model_code+"_mode_choice.csv", "a") as f:
            f.write("%s,%s,%s,%s,%.4f,%d,%.5f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % \
                (args.model_run_date, args.model_type, args.zoomlevel, "MNL", lr, -1, wd, 
                  train_kl, train_mse, train_mse1, train_mse2, train_mse3, train_mse4,
                  test_kl, test_mse, test_mse1, test_mse2, test_mse3, test_mse4,
                  r1, r2, r3, r4))
            
            
    trpgen_train =  y[~train_test_index,1]
    trpgen_test =  y[train_test_index,1]
    
    lr = linear_model.LinearRegression()
    lr.fit(x_train, trpgen_train)
    with open(out_dir+args.sampling[0]+"_"+args.model_code+"_regression_"+variable_names[-1]+".csv", "a") as f:
        f.write("%s,%s,%s,%.4f,%.4f,%.4f,%s,%s,%d,%d\n" % (args.model_run_date, args.model_type, variable_names[-1], -1, 
            lr.score(x_train, trpgen_train), lr.score(x_test, trpgen_test), 'lr', args.zoomlevel,
            np.sum(lr.coef_ != 0), len(lr.coef_)))
#     print(lr.score(x_train, trpgen_train), lr.score(x_test, trpgen_test))

    for a in np.linspace(0.005, 0.014, 10):
        lasso = linear_model.Lasso(alpha=a)
        lasso.fit(x_train, trpgen_train)
        with open(out_dir+args.sampling[0]+"_"+args.model_code+"_regression_"+variable_names[-1]+".csv", "a") as f:
            f.write("%s,%s,%s,%.6f,%.4f,%.4f,%s,%s,%d,%d\n" % (args.model_run_date, args.model_type, variable_names[-1], a, 
                lasso.score(x_train, trpgen_train), lasso.score(x_test, trpgen_test), 'lasso', args.zoomlevel,
                np.sum(lasso.coef_ != 0), len(lasso.coef_)))
    #     print(lasso.score(x_train, trpgen_train), lasso.score(x_test, trpgen_test))
    
    for a in np.linspace(1,4,10):
        ridge = linear_model.Ridge(alpha=a)
        ridge.fit(x_train, trpgen_train)
        with open(out_dir+args.sampling[0]+"_"+args.model_code+"_regression_"+variable_names[-1]+".csv", "a") as f:
            f.write("%s,%s,%s,%.4f,%.4f,%.4f,%s,%s,%d,%d\n" % (args.model_run_date, args.model_type, variable_names[-1], a, 
                ridge.score(x_train, trpgen_train), ridge.score(x_test, trpgen_test), 'ridge', args.zoomlevel,
                np.sum(ridge.coef_ != 0), len(ridge.coef_)))
    #     print(ridge.score(x_train, trpgen_train), ridge.score(x_test, trpgen_test))
    
    
