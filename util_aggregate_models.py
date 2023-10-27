import sys
sys.path.append("models/")
import itertools
import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn as nn

import mnl

from setup import out_dir

def mnl_torch(trainloader, testloader, n_features, sst_train, sst_test, lr_list, wd_list, save_models=False, save_name=''):
    kldivloss = nn.KLDivLoss(reduction='sum')
    mseloss = nn.MSELoss(reduction='none')

    all_dict = {}
    
    num_train = len(trainloader.dataset)
    num_test = len(testloader.dataset)
    
    for (lr, wd) in itertools.product(lr_list, wd_list):
        print(f"[lr: {lr:.2e}, wd: {wd:.2e}]")
        
        train_loss_list = []
        test_loss_list = []
        
        # model setup
        model = mnl.MNL(n_alts=4, n_features=n_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # model training
        ref1 = 0
        ref2 = 0
        
        for epoch in range(4000):

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

            train_kl = kl_/num_train
            train_mse = np.sqrt(mse_/num_train)
            train_mse1 = np.sqrt(mse1_/num_train)
            train_mse2 = np.sqrt(mse2_/num_train)
            train_mse3 = np.sqrt(mse3_/num_train)
            train_mse4 = np.sqrt(mse4_/num_train)

            train_r1 = 1-mse1_/sst_train[0]
            train_r2 = 1-mse2_/sst_train[1]
            train_r3 = 1-mse3_/sst_train[2]
            train_r4 = 1-mse4_/sst_train[3]

            loss_ = train_kl

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

                test_kl = kl_/num_test
                test_mse = np.sqrt(mse_/num_test)
                test_mse1 = np.sqrt(mse1_/num_test)
                test_mse2 = np.sqrt(mse2_/num_test)
                test_mse3 = np.sqrt(mse3_/num_test)
                test_mse4 = np.sqrt(mse4_/num_test)

#                 r1 = r2_score(y_batch.numpy()[:,0],torch.exp(probs).detach().numpy()[:,0])
#                 r2 = r2_score(y_batch.numpy()[:,1],torch.exp(probs).detach().numpy()[:,1])
#                 r3 = r2_score(y_batch.numpy()[:,2],torch.exp(probs).detach().numpy()[:,2])
#                 r4 = r2_score(y_batch.numpy()[:,3],torch.exp(probs).detach().numpy()[:,3])

                r1 = 1-mse1_/sst_test[0]
                r2 = 1-mse2_/sst_test[1]
                r3 = 1-mse3_/sst_test[2]
                r4 = 1-mse4_/sst_test[3]
            
                train_loss_list.append(train_kl)
                test_loss_list.append(test_kl)
                
                if epoch >= 40:
                    if (np.abs(loss_ - ref1)/ref1<0.001) & (np.abs(loss_ - ref2)/ref2<0.001):
                        print("Early stopping at epoch", epoch)
                        break
#                     if (ref1 < loss_) & (ref1 < ref2):
#                         print("Diverging. stop.")
#                         break
                    if loss_ < best:
                        best = loss_
                        best_epoch = epoch
                        output = (train_kl, train_mse, train_mse1, train_mse2, train_mse3, train_mse4,
                                  test_kl, test_mse, test_mse1, test_mse2, test_mse3, test_mse4,
                                  train_r1, train_r2, train_r3, train_r4,  r1, r2, r3, r4)
                else:
                    best = loss_
                    best_epoch = epoch
                    output = (train_kl, train_mse, train_mse1, train_mse2, train_mse3, train_mse4,
                                  test_kl, test_mse, test_mse1, test_mse2, test_mse3, test_mse4,
                                  train_r1, train_r2, train_r3, train_r4, r1, r2, r3, r4)
                ref2 = ref1
                ref1 = loss_

        print(f"[epoch: {best_epoch:>3d}] Train KL loss: {output[0]:.3f} Train R2 score: {output[12]:.3f} {output[13]:.3f} {output[14]:.3f} {output[15]:.3f} ")
        print(f"[epoch: {best_epoch:>3d}] Test KL loss: {output[6]:.3f} Test R2 score: {output[16]:.3f} {output[17]:.3f} {output[18]:.3f} {output[19]:.3f} ")
        print()
        
        if save_models:
            torch.save(model.state_dict(), out_dir+"mnl_models/"+save_name+"_"+str(lr)+"_"+str(wd)+".pt")
            
        ret_dict = {
            'train_kl_loss': output[0], 'test_kl_loss': output[6],
            'train_r2_auto': output[13], 'train_r2_active': output[12], 'train_r2_pt': output[15],
            'test_r2_auto': output[17], 'test_r2_active': output[16], 'test_r2_pt': output[19],
            'train_loss': train_loss_list, 'test_loss': test_loss_list,
        }
        
        all_dict[(lr,wd)] = ret_dict
        
    return all_dict