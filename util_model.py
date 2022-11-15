import sys
sys.path.append("models/")
import importlib
import pandas as pd
import torch
from collections import OrderedDict
from setup import out_dir
import argparse
import math

def parse_args(s=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_run_date', type=str, required=True)
    parser.add_argument('--zoomlevel', type=str, default='zoom13')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--color_channels', type=int, default=3)
    parser.add_argument('--demo_channels', type=int, default=0)    
    parser.add_argument('--data_version', type=str, default='1571')

    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default='200')
    parser.add_argument('--loss_func', type=str, default='mse')
    
    parser.add_argument('--tensorboard', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=True)

    parser.add_argument('--model_class', type=str, default='AE')
    parser.add_argument('--model_type', type=str, default='dcgan')
    parser.add_argument('--sampling', type=str, default='stratified')
    parser.add_argument('--normalization', type=str, default='minmax')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--im_norm', type=int, default=0)
    
    parser.add_argument('--v1', type=str, default='A')
    parser.add_argument('--v2', type=int, default='1')
    
    #     parser.add_argument('--lr_list', type=str, default="0.00005,0.0001,0.0005,0.001")
    #     parser.add_argument('--wd_list', type=str, default="0.001,0.005,0.01")
    #     parser.add_argument('--do_list', type=str, default="0,0.1,0.2,0.5")
    
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    
    if s is not None:
        args = parser.parse_args(s)
    else:
        args = parser.parse_args()
    
    if args.model_type == 'SAE':
        args.load_model_name = 'Autoencoder'
    elif args.model_type == 'AE':
        args.load_model_name = 'Autoencoder_raw'
    elif args.model_type == 'SAE_Adv':
        args.load_model_name = 'Autoencoder_adv'
       

    #     args.lr_list = [float(a) for a in args.lr_list.split(',')]
    #     args.wd_list = [float(a) for a in args.wd_list.split(',')]
    #     args.do_list = [float(a) for a in args.do_list.split(',')]

    return args

def sae_config(args):
        
    weight, lr, wd = get_hp_from_version_code(args.v1, args.v2)
    if weight >= 1:
        weightt = 1/weight
        weight = 1
    else:
        weightt = 1
        
    args = {'image_size': 224, 
        'depth': -1,
       'base_channels':64,
       'output_dim':args.output_dim,
       'num_demo_vars':10,#len(variable_names),
       'demo_norm': args.normalization,
       'weight': weight,
       'weightt': weightt,
       'cardinality':1,
       'epochs':400,
       'batch_size':16,
       'base_lr':lr,
       'weight_decay':wd,
       'momentum': 0.9,
       'nesterov': True,
       'milestones': '[50,100]',
       'lr_decay':0.1,
       'seed': 1234,
       'outdir':out_dir,
       'num_workers':8,
       'tensorboard':False,
       'save':True}

    model_config = OrderedDict([
        ('arch', 'resnext'),
        ('depth', args['depth']),
        ('base_channels', args['base_channels']),
        ('cardinality', args['cardinality']),
        ('input_shape', (1, 3, 32, 32)),
        ('output_dim', args['output_dim']),
        ('num_demo_vars', args['num_demo_vars'])
    ])

    optim_config = OrderedDict([
        ('epochs', args['epochs']),
        ('batch_size', args['batch_size']),
        ('base_lr', args['base_lr']),
        ('weight_decay', args['weight_decay']),
        ('momentum', args['momentum']),
        ('nesterov', args['nesterov']),
        ('milestones', json.loads(args['milestones'])),
        ('lr_decay', args['lr_decay']),
    ])

    data_config = OrderedDict([
        ('dataset', 'CIFAR10'),
        ('image_size', args['image_size']),
        ('demo_norm', args['demo_norm'])
    ])

    run_config = OrderedDict([
        ('weight', args['weight']),
        ('weightt', args['weightt']),
        ('seed', args['seed']),
        ('outdir', args['outdir']),
        ('save', args['save']),
        ('num_workers', args['num_workers']),
        ('tensorboard', args['tensorboard']),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])
    
    return config


def dcgan_config(args):
    
    model_config = OrderedDict([
        ('model_class', args.model_class),
        ('arch', 'resnet'),
        ('latent_dim', args.latent_dim),
        ('layers', [3,4,6,3]),# args['layers'])
        ('base_channels', 64),
        ('loss_func', args.loss_func)
    ])

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', 64), #args['batch_size']),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
    ])

    data_config = OrderedDict([
        ('image_size', args.image_size),
        ('color_channels', args.color_channels),
        ('demo_channels', args.demo_channels),
        ('im_norm', args.im_norm)
    ])

    run_config = OrderedDict([
        ('outdir', out_dir),
        ('save', args.save),
        ('num_workers', 8), #args['num_workers']),
        ('tensorboard', args.tensorboard),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])
    
    return config


def resnext_config(args):
    if args.latent_dim == -1:
        ld = args.output_dim**2*2048
    else:
        ld = args.latent_dim

    model_config = OrderedDict([
        ('model_class', args.model_class),
        ('arch', 'resnext'),
        ('base_channels', 64),
        ('cardinality', 1),
        ('output_dim', args.output_dim),
        ('latent_dim', ld)
    ])

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', 64),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
    ])

    data_config = OrderedDict([
        ('image_size', args.image_size),
        ('color_channels', args.color_channels),
        ('im_norm', args.im_norm)
    ])

    run_config = OrderedDict([
        ('save', args.save),
        ('num_workers', 8),
        ('tensorboard', args.tensorboard),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])
    
    return config

def load_model(script, model, config):
    module = importlib.import_module(script)
    Network = getattr(module, model)
    return Network(config)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count
        
def get_layers(model: torch.nn.Module):
    # get layers from model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_layers(child))
            except TypeError:
                flatt_children.append(get_layers(child))
    return flatt_children
    
def my_loss(out_image, out_demo, data, census_data, factor=1, factorr=1, return_components=False):
    reconstruct_loss = torch.mean((out_image - data)**2)
    regression_loss = torch.mean((out_demo - census_data)**2)
#     print(reconstruct_loss, regression_loss)
    if return_components:
        return reconstruct_loss, regression_loss
    else:
        return reconstruct_loss * factorr + regression_loss * factor

def adv_loss(out_image, out_demo, out_demo_adv, data, census_data, factor=10, return_components=False):
    reconstruct_loss = torch.mean((out_image - data)**2)
    regression_loss = torch.mean((out_demo - census_data)**2)
    adv_loss = torch.mean(torch.abs((out_demo * out_demo_adv).sum(-1)))
#     print(reconstruct_loss.item(), regression_loss.item(), adv_loss.item())
    if return_components:
        return reconstruct_loss, regression_loss, adv_loss
    else:
        return reconstruct_loss + regression_loss + adv_loss

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor, reduction='mean') -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    if reduction == 'mean':
        return torch.mean(_log_cosh(y_pred - y_true))
    elif reduction == 'sum':
        return torch.sum(_log_cosh(y_pred - y_true))
