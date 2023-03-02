import sys
sys.path.append("models/")
import itertools
import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn as nn

import mnl

from setup import out_dir


def train_mnl(trainloader, testloader, dim_embed, dim_demo, lr_list, wd_list, save_models=False, save_name=''):
    
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    all_dict = {}
    num_train = len(trainloader.dataset)
    num_test = len(testloader.dataset)
 
    for (lr, wd) in itertools.product(lr_list, wd_list):
        print(f"[lr: {lr:.2e}, wd: {wd:.2e}]")

        model = mnl.MNL2(n_alts=4, dim_embed=dim_embed, dim_demo=dim_demo)
        optimizer = torch.optim.Adam([{'params': model.mnl_embedding.parameters(), 'weight_decay': wd},
                                  {'params': model.mnl_demo.parameters(), 'weight_decay': wd/500}], lr=lr)

        ref1 = 0
        ref2 = 0

        for epoch in range(500):
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
                loss_ /= num_train
                train_acc = correct/num_train
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
                test_acc = correct/num_test
                print(f"[epoch: {epoch:>3d}] Test loss: {test_loss_:.4f} accuracy: {test_acc:.3f}")

                if epoch > 15:
                    if (np.abs(loss_ - ref1)/ref1<0.001) & (np.abs(loss_ - ref2)/ref2<0.001):
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
                
            return_dict = {'train_loss': loss_, 'test_loss': test_loss_, 'train_acc': train_acc, 'test_acc': test_acc}
            
        all_dict[(lr,wd)] = return_dict
    
    return all_dict

                