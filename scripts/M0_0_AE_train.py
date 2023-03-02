import sys
sys.path.append("models/")
from setup import out_dir, data_dir, image_dir, model_dir

import os
from datetime import datetime
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import glob
import pandas as pd

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torchvision.transforms
import torch.nn.functional as F

from util_model import parse_args, dcgan_config, resnext_config, log_cosh_loss
import util_image

try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from dataloader import image_loader
from autoencoder import Autoencoder_raw
from M0_0_util_train_test import load_model, train, test

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    
    logging.basicConfig(
        format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    args = parse_args(['@scripts/args_autoencoder.txt'])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")    
    
    if args.model_arch == 'dcgan':
        
        config = dcgan_config(args)
        encoder = load_model('dcgan', 'Discriminator', config)
        decoder = load_model('dcgan', 'Generator', config)
        
    elif args.model_arch == 'resnet':
        
        config = dcgan_config(args)
        encoder = load_model('resnet', 'Encoder', config)
        decoder = load_model('resnet', 'Decoder', config)

    elif args.model_arch == 'resnet1':
        
        config = dcgan_config(args)
        encoder = load_model('resnet1', 'ResNet18Enc', config)
        decoder = load_model('resnet1', 'ResNet18Dec', config)
        
    elif args.model_arch == 'resnext':
        
        config = resnext_config(args)
        encoder = load_model(config['model_config']['arch'], 'Encoder', config)
        decoder = load_model(config['model_config']['arch'], 'Decoder', config)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    config['encoder'] = encoder
    config['decoder'] = decoder

    model = load_model('autoencoder','Autoencoder_raw', config)
    model = model.to(device)

    n_params = sum([param.view(-1).size()[0] for param in encoder.parameters()]) + \
               sum([param.view(-1).size()[0] for param in decoder.parameters()])

    print('n_params: {}'.format(n_params))
    
    model_config = config['model_config']
    run_config = config['run_config']
    data_config = config['data_config']
    optim_config = config['optim_config']
    
    if args.loss_func == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    elif args.loss_func == 'cosh':
        criterion = log_cosh_loss
    else:
        print("Loss func error!")
        
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = optim_config['base_lr'],
        weight_decay = optim_config['weight_decay'],
        )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    
    # TensorBoard SummaryWriter
    writer = SummaryWriter(model_name) if run_config['tensorboard'] else None
    
    train_loader, test_loader = image_loader(image_dir+args.zoomlevel+"/", data_dir, optim_config['batch_size'], 
         run_config['num_workers'], 
         data_config['image_size'], 
         data_version=args.data_version, 
         sampling=args.sampling, 
         recalculate_normalize=False,
         augment=False,
         norm=args.im_norm)
    
    ref1 = 0
    ref2 = 0

    train_loss_list = []
    test_loss_list = []

    train_flag = True
    
    for epoch in range(optim_config['epochs']):

        loss_ = train(epoch, model, optimizer, criterion, train_loader, run_config,
             writer, device, logger=logger)
        train_loss_list.append(loss_)

        scheduler.step()

        test_loss_ = test(epoch, model, criterion, test_loader, run_config,
                        writer, device, logger, return_output=False)
        test_loss_list.append(test_loss_)

        if epoch % 5 == 0:
            
            test_image_recon = test(epoch, model, criterion, test_loader, run_config, 
                                writer, device, logger, return_output=True)
            if args.im_norm != 0:

                if args.im_norm == 1:
                    if args.zoomlevel == 'zoom13':
                        mean = [0.3733, 0.3991, 0.3711]
                        std = [0.2173, 0.2055, 0.2143]
                    elif args.zoomlevel == 'zoom15':
                        mean = [0.3816, 0.4169, 0.3868]
                        std = [0.1960, 0.1848, 0.2052]

                elif args.im_norm == 2:
                    mean = [0.5,0.5,0.5]
                    std = [0.5,0.5,0.5]

                test_image_recon = util_image.inverse_transform(test_image_recon, mean, std)

            fig, ax = plt.subplots(1, 2, figsize=(4,2))
            ax[0].imshow(test_image_recon[0,:,:,:].permute(1,2,0).detach().cpu().numpy())
            ax[0].axis('off')
            ax[1].imshow(test_image_recon[9,:,:,:].permute(1,2,0).detach().cpu().numpy())
            ax[1].axis('off')
            fig.savefig(out_dir+'AE_Recon/64_'+str(epoch)+".png", bbox_inches='tight')    
        
    
            if epoch > 50:
                if (np.abs(loss_ - ref1)/ref1<0.001) & (np.abs(loss_ - ref2)/ref2<0.001):
                    print("Early stopping at epoch", epoch)
                    break
                if (ref1 < loss_) & (ref1 < ref2):
                    print("Diverging. stop.")
                    train_flag = False
                    break
                if loss_ < best:
                    best = loss_
                    best_test = test_loss_
                    best_epoch = epoch
            else:
                best = loss_
                best_test = test_loss_
                best_epoch = epoch

            ref2 = ref1
            ref1 = loss_

            if (config['run_config']['save']) & (best_epoch==epoch):
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': best,
                    'test_loss': best_test,
                    'config': config},
                    model_dir+"Autoencoder/"+args.loss_func+"_"+args.zoomlevel+"_"+str(model_config['latent_dim'])+"_"+str(args.image_size)+"_"+str(int(args.im_norm))+"_"+str(args.model_run_date)+"_"+str(epoch)+".pt")

    if config['run_config']['save']:
        files = glob.glob(model_dir+"Autoencoder/"+args.loss_func+"_"+args.zoomlevel+"_"+str(model_config['latent_dim'])+"_"+str(args.image_size)+"_"+str(int(args.im_norm))+"_"+str(args.model_run_date)+"_*.pt")    
        
        for f in files:
            e = int(f.split("_")[-1].split(".")[0])
            if e != best_epoch:
                os.remove(f)

    with open(out_dir+"AE_tries.csv", "a") as f:
        f.write("%s,%s,%d,%d,%d,%.2E,%.2E,%d,%.4f,%.4f,%d,%s,%s\n" % (args.model_run_date, args.zoomlevel, model_config['latent_dim'], args.image_size, args.im_norm, args.base_lr, args.weight_decay, best_epoch, best, best_test, train_flag, args.loss_func, args.model_arch))
    
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(train_loss_list, color='cornflowerblue', label='Train')
    ax.plot(test_loss_list, color='sandybrown', label='Test')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim([0, 1.1*np.max(train_loss_list+test_loss_list)])
    ax.legend()
    fig.savefig(out_dir+"training_plots/"+args.loss_func+"_"++args.zoomlevel+"_"+str(model_config['latent_dim'])+"_"+str(args.image_size)+"_"+str(int(args.im_norm))+"_"+str(args.model_run_date)+".png", bbox_inches='tight')

#     if run_config['tensorboard']:
#         outpath = os.path.join(outdir, 'all_scalars.json')
#         writer.export_scalars_to_json(outpath)
    
