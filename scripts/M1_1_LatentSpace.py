import sys
sys.path.append("models/")
from setup import proj_dir, out_dir, data_dir, image_dir, model_dir, parse_args, configuration

import numpy as np
import random
import glob
import pandas as pd
import pickle as pkl
import torch

from dataloader import get_loader, image_loader, load_demo
from autoencoder import Autoencoder
from M1_util_train_test import load_model, train, test, AverageMeter
from util_model import my_loss

if __name__ == "__main__":
    args = parse_args()
    config = configuration(args)
    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']
    model_config = config['model_config']
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Check one model exists for this config
    model_path = glob.glob(model_dir+args.model_type+"_"+args.zoomlevel+"_"+str(args.output_dim**2*2048)+"_"+
                           args.model_run_date+"_*.pt")
    #
    if len(model_path) == 1:
        saved = torch.load(model_path[0])
        print(model_path[0], "loaded.")
    else:
        print("Error. More than one model or no model exists.")
        print(model_path)
        print(model_dir+args.model_type+"_"+args.zoomlevel+"_"+str(args.output_dim**2*2048)+"_"+
                           args.model_run_date+"_*.pt")
        
    # load model
    config['model_config']['input_shape'] = (1,3,data_config['image_size'],data_config['image_size'])

    encoder = load_model(config['model_config']['arch'], 'Encoder', config['model_config'])

    config['model_config']['input_shape'] = [1,2048,config['model_config']['output_dim'],config['model_config']['output_dim']]

    config['model_config']['conv_shape'] = [data_config['image_size']//32,data_config['image_size']//32]
    config['model_config']['output_channels'] = 3

    decoder = load_model(config['model_config']['arch'], 'Decoder', config['model_config'])

    config['encoder'] = encoder
    config['decoder'] = decoder

    model = load_model('autoencoder',args.load_model_name, config)
    model.load_state_dict(saved['model_state_dict']);

    model = model.to(device)
    
    train_loader, test_loader = image_loader(image_dir+args.zoomlevel+"/", data_dir, optim_config['batch_size'], run_config['num_workers'], 
                                         data_config['image_size'], 
                                         sampling=args.sampling, recalculate_normalize=False)

    
    # Check if embeddings have been saved
    files = glob.glob(proj_dir+"latent_space/"+args.model_type+"_"+args.zoomlevel+"_"+str(args.output_dim**2*2048)+"_"+
                           args.model_run_date+".pkl")

    if len(files) == 1:
        print("Loading Existing Embedding", proj_dir+"latent_space/"+args.model_type+"_"+args.zoomlevel+"_"+str(args.output_dim**2*2048)+"_"+
                           args.model_run_date+".pkl")
        with open(proj_dir+"latent_space/"+args.model_type+"_"+args.zoomlevel+"_"+str(args.output_dim**2*2048)+"_"+
                           args.model_run_date+".pkl", "rb") as f:
            encoder_output = pkl.load(f)
            im = pkl.load(f)
            ct = pkl.load(f)

    elif len(files) > 1:
        print("Multiple Files exist. Check specified path.")

    else: # Calculate Embedding
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

        # Save Embeddings

        with open(proj_dir+"latent_space/"+args.model_type+"_"+args.zoomlevel+"_"+str(args.output_dim**2*2048)+"_"+
                               args.model_run_date+".pkl", "wb") as f:
            pkl.dump(encoder_output, f)
            pkl.dump(im, f)
            pkl.dump(ct, f)
            
    print()

