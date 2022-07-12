import sys
sys.path.append("models/")
from setup import proj_dir, out_dir, data_dir, image_dir, model_dir, parse_args, configuration

import os
from datetime import datetime
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import pandas as pd
import pickle as pkl
import torch
from torch.utils.tensorboard import SummaryWriter

from dataloader import image_loader, load_demo
from autoencoder import Autoencoder
from M1_util_train_test import load_model, train, test, AverageMeter
from util_model import my_loss

if __name__ == "__main__":

    logging.basicConfig(
        format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    args = parse_args()
    config = configuration(args)
    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']
    model_config = config['model_config']
    
    # tensorboard 
    writer = SummaryWriter(proj_dir+'tensorboard/runs/'+args.model_type+"_"+args.zoomlevel+"_"+str(args.output_dim**2*2048)+"_"+
                           args.model_run_date+"/")
    
    variable_names = ['tot_population','pct25_34yrs','pct35_50yrs','pctover65yrs',
             'pctwhite_alone','pct_nonwhite','pctblack_alone',
             'pct_col_grad','avg_tt_to_work','inc_per_capita']
    model_save_variable_names = ['totpop','pct25-34','pct35-50','pctsenior',
             'pctwhite_alone','pct_nonwhite','pctblack_alone',
             'pctcolgrad','avg_tt_to_work','inc']
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # data loaders
    demo_cs, demo_np = load_demo(data_dir, norm=args.normalization)
    train_loader, test_loader = image_loader(image_dir+args.zoomlevel+"/", data_dir, optim_config['batch_size'], 
                                         run_config['num_workers'],
                                         data_config['image_size'], data_version=args.data_version, sampling=args.sampling, 
                                         recalculate_normalize=False)
    
    criterion = my_loss

    # model
    config['model_config']['input_shape'] = (1,3,data_config['image_size'],data_config['image_size'])

    encoder = load_model(config['model_config']['arch'], 'Encoder', config['model_config'])
    encoder = encoder.to(device)

    config['model_config']['input_shape'] = [1,2048,config['model_config']['output_dim'],config['model_config']['output_dim']]

    config['model_config']['conv_shape'] = [data_config['image_size']//32,data_config['image_size']//32]
    config['model_config']['output_channels'] = 3

    decoder = load_model(config['model_config']['arch'], 'Decoder', config['model_config'])
    decoder = decoder.to(args.device)

    config['encoder'] = encoder
    config['decoder'] = decoder
    model = load_model('autoencoder','Autoencoder', config)
    model = model.to(device)

    n_params = sum([param.view(-1).size()[0] for param in encoder.parameters()]) +\
               sum([param.view(-1).size()[0] for param in decoder.parameters()])
    logger.info('n_params: {}'.format(n_params))

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optim_config['base_lr'],
        momentum=optim_config['momentum'],
        weight_decay=optim_config['weight_decay'],
        nesterov=optim_config['nesterov'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=optim_config['milestones'],
        gamma=optim_config['lr_decay'])

    ref1 = 0
    ref2 = 0
    
    train_loss_list = []
    test_loss_list = []
    train_flag = True
    
    for epoch in range(optim_config['epochs']):

        loss_ = train(epoch, model, optimizer, criterion, train_loader, (demo_cs,demo_np), run_config,
             writer, device, logger=logger)
        train_loss_list.append(loss_)

        test_loss_ = test(epoch, model, criterion, test_loader, (demo_cs,demo_np), run_config,
                        writer, device, logger, return_output=False)
        test_loss_list.append(test_loss_)
        
        writer.add_scalar("Loss/Train", loss_)
        writer.add_scalar("Loss/Test", test_loss_)
	
        if epoch % 5 == 0:
            if epoch > 50:
                if (np.abs(loss_ - ref1)/ref1<0.0001) & (np.abs(loss_ - ref2)/ref2<0.0001):
                    print("Early stopping at epoch", epoch)
                    break
                if (ref1 < loss_) & (ref1 < ref2):
                    print("Diverging. stop.")
                    train_flag = False
                    break
                if loss_ < best:
                    best = loss_
                    best_epoch = epoch
            else:
                best = loss_
                best_epoch = epoch

            ref2 = ref1
            ref1 = loss_

            if (config['run_config']['save']) & (best_epoch==epoch):
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config},
                    model_dir+"SAE_"+args.zoomlevel+"_"+str(model_config['output_dim']**2*2048)+"_"+
                    args.model_run_date+"_"+str(epoch)+".pt")


    if config['run_config']['save']:
        files = glob.glob(model_dir+"SAE_"+args.zoomlevel+"_"+str(model_config['output_dim']**2*2048)+"_"+
                                  args.model_run_date+"_*.pt")

        for f in files:
            e = int(f.split("_")[-1].split(".")[0])
            if e != best_epoch:
                os.remove(f)

    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(train_loss_list, color='cornflowerblue', label='Train')
    ax.plot(test_loss_list, color='sandybrown', label='Test')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim([0, 1.1*np.max(train_loss_list+test_loss_list)])
    ax.legend()
    fig.savefig(out_dir+"training_plots/SAE_"+args.zoomlevel+"_"+str(model_config['output_dim']**2*2048)+"_"+
                                  args.model_run_date+".png", bbox_inches='tight')

    model.eval()
    loss_meter_1 = AverageMeter()
    loss_meter_2 = AverageMeter()

    for step, (image_list, data) in enumerate(test_loader):

        census_index = [demo_cs.index(i[i.rfind('/')+1:i.rfind('_')]) for i in image_list]
        census_data = demo_np[census_index]

        census_data = torch.tensor(census_data).to(device)
        data = data.to(device)

        out_image, out_demo = model(data)

        loss1, loss2 = criterion(out_image, out_demo, data, census_data, return_components=True)

        num = data.size(0)

        loss_meter_1.update(loss1.item(), num)
        loss_meter_2.update(loss2.item(), num)

#         if step % 10 == 0:
#             print(step, end='\t')

    best_test_1 = loss_meter_1.avg
    best_test_2 = loss_meter_2.avg
    print(best_test_1, best_test_2)         

    loss_meter_1 = AverageMeter()
    loss_meter_2 = AverageMeter() 
    
    for step, (image_list, data) in enumerate(train_loader):

        census_index = [demo_cs.index(i[i.rfind('/')+1:i.rfind('_')]) for i in image_list]
        census_data = demo_np[census_index]

        census_data = torch.tensor(census_data).to(device)
        data = data.to(device)

        out_image, out_demo = model(data)

        loss1, loss2 = criterion(out_image, out_demo, data, census_data, return_components=True)

        num = data.size(0)

        loss_meter_1.update(loss1.item(), num)
        loss_meter_2.update(loss2.item(), num)

#         if step % 10 == 0:
#             print(step, end='\t')

    best_1 = loss_meter_1.avg
    best_2 = loss_meter_2.avg
    print(best_1, best_2)
    
    with open(out_dir+"SAE_train.csv", "a") as f:
        f.write("%s,%s,%d,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%d\n" % (args.model_run_date, args.zoomlevel, model_config['output_dim']**2*2048, args.sampling, args.normalization, best_epoch, best_1, best_2, best_test_1, best_test_2, train_flag))

    # Save Embeddings
    ct = []
    encoder_output = []
    im = []

    for step, data in enumerate(train_loader):
        data1 = data[1].to(device)
        ct += [s[s.rindex("/")+1: s.rindex("_")]for s in data[0]]
        encoder_output += [encoder(data1).cpu().detach().numpy()]
        im += data[0]
        if step % 10 == 0:
            print(step, end='\t')

    for step, data in enumerate(test_loader):
        data1 = data[1].to(device)
        ct += [s[s.rindex("/")+1: s.rindex("_")]for s in data[0]]
        encoder_output += [encoder(data1).cpu().detach().numpy()]
        im += data[0]
        if step % 10 == 0:
            print(step, end='\t')

    encoder_output = np.vstack(encoder_output)    

#     print(encoder_output.shape)
    encoder_output = encoder_output.reshape(len(encoder_output), -1)

    with open(proj_dir+"latent_space/"+args.model_type+"_"+args.zoomlevel+"_"+str(args.output_dim**2*2048)+"_"+
                           args.model_run_date+".pkl", "wb") as f:
        pkl.dump(encoder_output, f)
        pkl.dump(im, f)
        pkl.dump(ct, f)
    
    print()
    
