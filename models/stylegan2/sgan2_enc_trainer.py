import os
import sys
sys.path.append("models/")
import math
import json
from datetime import datetime

from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np

import torch
from torch import nn, einsum
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
from torchvision import transforms
import lpips

from sgan2 import StyleGAN2
import dcgan
from ghfeat import GHFeat_Enc
from stylex import DiscriminatorE

from util import *
import aim

from PIL import Image
from pathlib import Path

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

sys.path.append("../../")
from dataloader import ImageDataset, ImageHDF5
from setup import data_dir, image_dir


# Our losses
l1_loss = nn.L1Loss()
kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
mse_loss = nn.MSELoss()

def reconstruction_loss(encoder_batch: torch.Tensor, generated_images: torch.Tensor, 
                        generated_images_w: torch.Tensor, encoder_w: torch.Tensor, lpips_loss):
    
    # images normalized to [-1,+1] for lpips loss
    encoder_batch_norm = lpips_normalize(encoder_batch)
    generated_images_norm = lpips_normalize(generated_images)

    # LPIPS reconstruction loss
    loss1 = 0.1 * lpips_loss(encoder_batch_norm, generated_images_norm).mean() 
    loss2 = 0.1 * l1_loss(encoder_w, generated_images_w) 
    loss3 = 1 * l1_loss(encoder_batch, generated_images)
    # loss = loss1 + loss2 + loss3
#     loss = loss1 + loss3
#     loss = loss2 + loss3
       
    return loss1, loss2, loss3    
    
    
# constants

NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new
    
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)
    

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)
    
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x
    
    
class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))
    
    
class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, *_, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss
    
    
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)

    
# helpers
def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def default(value, d):
    return value if exists(value) else d

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    return el if isinstance(el, list) else [el]

def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts =  head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

def noise(n, latent_dim, device, demo_batch=None):
    if demo_batch is None:
        return torch.randn(n, latent_dim).cuda(device)
    else:
        return torch.cat([torch.randn(n, latent_dim-demo_batch.shape[1]), demo_batch], dim=1).cuda(device)
    
def noise_list(n, layers, latent_dim, device, demo_batch=None):
    return [(noise(n, latent_dim, device, demo_batch), layers)]

def mixed_list(n, layers, latent_dim, device, demo_batch=None):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device, demo_batch) + noise_list(n, layers - tt, latent_dim, device, demo_batch)

def latent_to_w(style_vectorizer, latent_descr, demo_batch=None):
    if demo_batch is None:
        return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]
    else:
        assert len(latent_descr) == len(demo_batch)
        ret = []
        for i in range(len(latent_descr)):
            ret.append((torch.cat([style_vectorizer(latent_descr[i][0]), demo_batch[i]], dim=1).float(), latent_descr[i][1]))
#         return [(torch.cat([style_vectorizer(z), demo], dim=1).float(), num_layers) for (z, num_layers),demo in zip(latent_descr,demo_batch)]
        return ret
    
def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


# losses

def gen_hinge_loss(fake, real):
    return fake.mean()

def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)



# augmentations

def random_hflip(tensor, prob):
    if prob < random():
        return tensor
    return torch.flip(tensor, dims=(3,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)
    
class Trainer():
    def __init__(
        self,
        stylegan_name = 'default',
        encoder_name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        image_size = 128,
        conditional = False,
        encoder_class = '',
        
        network_capacity = 16,
        fmap_max = 512,
        transparent = False,
        batch_size = 4,
        mixed_prob = 0.9,
        gradient_accumulate_every=1,
        lr = 2e-4,
        lr_mlp = 0.1,
        ttur_mult = 2,
        rel_disc_loss = False,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 1000,
        num_image_tiles = 8,
        trunc_psi = 0.6,
        fp16 = False,
        cl_reg = False,
        no_pl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        aug_prob = 0.,
        aug_types = ['translation', 'cutout'],
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        dual_contrast_loss = False,
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        rec_scaling = 1,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None
        self.encoder = None

        self.stylegan_name = stylegan_name
        self.encoder_name = encoder_name
        
        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = Path(results_dir) / 'sGAN2'/ self.stylegan_name / self.encoder_name
        self.models_dir = Path(models_dir) / 'sGAN2'/ self.stylegan_name
#         self.fid_dir = base_dir / 'fid' / self.stylegan_name
        self.config_path = self.models_dir / self.encoder_name / 'config.json'
        
        print("Results directory:", self.results_dir )
        print("Model directory:", self.models_dir )
        
        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.conditional = conditional
        if self.conditional:
            self.demo_channels = kwargs['demo_channels']
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.num_image_tiles = num_image_tiles
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        self.dual_contrast_loss = dual_contrast_loss

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.is_ddp = is_ddp
        ## Changed to 1 by default since no multi-gpu processing is happening
        ## Need to change back if multiple gpus are used.
#         self.is_main = rank == 0
        self.is_main = True
        self.rank = rank
        self.world_size = world_size

        self.encoder_class = encoder_class
        self.rec_scaling = rec_scaling
        
        self.lpips_loss = lpips.LPIPS(net="alex").cuda(self.rank) # image should be RGB, IMPORTANT: normalized to [-1,1]
        self.tb_writer = None
        
        self.logger = {}

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}
        
    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr = self.lr, lr_mlp = self.lr_mlp, ttur_mult = self.ttur_mult, image_size = self.image_size, conditional = self.conditional, network_capacity = self.network_capacity, fmap_max = self.fmap_max, transparent = self.transparent, fq_layers = self.fq_layers, fq_dict_size = self.fq_dict_size, attn_layers = self.attn_layers, fp16 = self.fp16, cl_reg = self.cl_reg, no_const = self.no_const, rank = self.rank, *args, **kwargs)

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank]}
            self.S_ddp = DDP(self.GAN.S, **ddp_kwargs)
            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    def init_encoder(self):
        
        if self.encoder_class is None:
            self.encoder = DiscriminatorE(image_size, network_capacity, encoder=True, fq_layers=fq_layers,
                                          fq_dict_size=fq_dict_size,
                                          attn_layers=attn_layers, transparent=transparent, fmap_max=fmap_max,
                                          encoder_dim=latent_dim-1)
        elif self.encoder_class == 'DCGAN':
            model_config = {"base_channels":64, "model_class":"StylEx", "latent_dim":latent_dim-1}
            data_config = {"color_channels":3}
            config = {"model_config": model_config, "data_config": data_config}
            
            self.encoder = dcgan.Discriminator(config)
            self.encoder.encoder_dim = model_config['latent_dim']
        elif self.encoder_class == 'GHFeat':
            # right now, only works with latent dim = 512
            self.encoder = GHFeat_Enc(self.image_size)
        
        self.encoder = self.encoder.cuda(self.rank)
        self.E_opt = Adam(self.encoder.parameters(), lr=self.lr, betas=(0.5, 0.9))

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.fmap_max = config.pop('fmap_max', 512)
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.lr_mlp = config.pop('lr_mlp', 0.1)


    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp, 'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, folder):
        
#         self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
#         transform = torchvision.transforms.Compose([
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.CenterCrop(224),
#             torchvision.transforms.Resize(self.image_size)])
#             torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        transform=None
#         self.dataset = ImageDataset(folder, data_dir, train=True, data_version='1571', transform=transform, sampling='clustered', image_type='png', augment=None, demo=self.conditional)
        self.dataset = ImageHDF5(folder, data_dir, train=True, transform=transform)
        num_workers = default(self.num_workers, NUM_CORES if not self.is_ddp else 0)
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def set_test_data_src(self, folder, batch_size):
        
#         transform = transform = torchvision.transforms.Compose([
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.CenterCrop(224),
#             torchvision.transforms.Resize(self.image_size)])
        transform=None
#         self.test_dataset = ImageDataset(folder, data_dir, train=False, data_version='1571', transform=transform, sampling='clustered', image_type='png', augment=None, demo=self.conditional)
        self.test_dataset = ImageHDF5(folder, data_dir, train=False, transform=transform)

        # num_workers = num_workers = default(self.num_workers, NUM_CORES if not self.is_ddp else 0)
        self.test_dataloader = data.DataLoader(self.test_dataset, num_workers = 0, batch_size = batch_size, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        
      
    def train_encoder_only(self, train=True, log=True):
                
        assert exists(self.GAN), 'Must load trained StylEx G, D, and S to train encoder only'
        
        if not exists(self.encoder):
            self.init_encoder()
            
        self.encoder.train()
        
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_rec_loss = torch.tensor(0.).cuda(self.rank)
        total_l1 = torch.tensor(0.).cuda(self.rank)
        total_l2 = torch.tensor(0.).cuda(self.rank)
        total_l3 = torch.tensor(0.).cuda(self.rank)

        batch_size = math.ceil(self.batch_size / self.world_size)
        
        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob = self.aug_prob
        aug_types = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}
        
        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        backwards = partial(loss_backwards, self.fp16)
        
        D_loss_fn = hinge_loss
        G_loss_fn = gen_hinge_loss
        
        self.E_opt.zero_grad()
        self.GAN.D_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug]):
            _,image_batch = next(self.loader)
            image_batch = torch.flatten(image_batch, start_dim=0, end_dim=-4)
            batch_size = len(image_batch)
            image_batch = image_batch.cuda(self.rank)
            image_batch.requires_grad_()
            
#             discriminator_batch = next(self.loader).cuda(self.rank)
#             discriminator_batch.requires_grad_()

            encoder_output = self.encoder(image_batch)
#             real_classified_logits = self.classifier(image_batch)
#             style = [(torch.cat((encoder_output, real_classified_logits), dim=1),
#                       self.GAN.G.num_layers)]  # Has to be bracketed because expects a noise mix
            if self.encoder_class != 'GHFeat':
                style = [(encoder_output, self.GAN.G.num_layers)]
                w_styles = styles_def_to_tensor(style)
            else:
                w_styles = encoder_output
                
            noise = image_noise(batch_size, image_size, device=self.rank)

            
            generated_images = G(w_styles, noise)
            fake_output, fake_q_loss = D_aug(generated_images.clone().detach(), detach=True, **aug_kwargs)

            real_output, real_q_loss = D_aug(image_batch, **aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output

            if self.rel_disc_loss:
                real_output_loss = real_output_loss - fake_output.mean()
                fake_output_loss = fake_output_loss - real_output.mean()

            divergence = D_loss_fn(real_output_loss, fake_output_loss)
            disc_loss = divergence

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                self.track(self.last_gp_loss, 'GP')
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, loss_id=1, retain_graph=True)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')

        if train:
            self.GAN.D_opt.step()

 
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[]):
            _,image_batch = next(self.loader)
            image_batch = torch.flatten(image_batch, start_dim=0, end_dim=-4)
            batch_size = len(image_batch)
            image_batch = image_batch.cuda(self.rank)
            image_batch.requires_grad_()
            
            real_output, real_q_loss = D_aug(image_batch, **aug_kwargs)

            encoder_output = self.encoder(image_batch)
#             real_classified_logits = self.classifier(image_batch)

#             style = [(torch.cat((encoder_output, real_classified_logits), dim=1), self.GAN.G.num_layers)]
            if self.encoder_class != 'GHFeat':
                style = [(encoder_output, self.GAN.G.num_layers)]
                w_styles = styles_def_to_tensor(style)
            else:
                w_styles = encoder_output

            noise = image_noise(batch_size, image_size, device=self.rank)
            
            generated_images = self.GAN.G(w_styles, noise)

            fake_output, fake_q_loss = D_aug(generated_images, **aug_kwargs)
            fake_output_loss = fake_output

            # Our losses
            # multiply losses by 2 since they are only calculated every other iteration if using alternating training
            # but we are not doing that, so we do not need to multiply by 2
            # we also do not need rec_rescaling since we dont have other losses
            l1, l2, l3 = reconstruction_loss(image_batch, generated_images, self.encoder(generated_images),
                                                          encoder_output, self.lpips_loss)
            l1 = l1 / self.gradient_accumulate_every
            l2 = l2 / self.gradient_accumulate_every
            l3 = l3 / self.gradient_accumulate_every
            rec_loss = self.rec_scaling * (l1+l2+l3)
                        
            gen_disc_loss = G_loss_fn(fake_output, real_output) / self.gradient_accumulate_every

#                 rec_loss.cuda(self.rank)
#                 kl_loss = 2 * self.kl_scaling * classifier_kl_loss(real_classified_logits,
#                                                                    gen_image_classified_logits) / self.gradient_accumulate_every

#             if apply_path_penalty:
#                 pl_lengths = calc_pl_lengths(w_styles, generated_images)
#                 avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

#                 if not is_empty(self.pl_mean):
#                     pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
#                     if not torch.isnan(pl_loss):
#                         gen_loss = gen_loss + pl_loss

#             gen_loss = gen_loss / self.gradient_accumulate_every
#             gen_loss.register_hook(raise_if_nan)

            backwards(gen_disc_loss, self.E_opt, loss_id=3, retain_graph=True)
            backwards(rec_loss, self.E_opt, loss_id=2, retain_graph=True)
#             backwards(kl_loss, self.GAN.G_opt, loss_id=3)

            total_l1 += l1.detach().item()
            total_l2 += l2.detach().item()
            total_l3 += l3.detach().item()

            total_rec_loss = total_l1 + total_l2 + total_l3
#             total_kl_loss += kl_loss.detach().item()
            total_gen_disc_loss += gen_disc_loss.detach().item() / self.gradient_accumulate_every

            self.total_rec_loss = float(total_rec_loss)
#             self.total_kl_loss = float(total_kl_loss)
            self.total_l1 = float(total_l1)
            self.total_l2 = float(total_l2)
            self.total_l3 = float(total_l3)
            self.total_gen_disc_loss = float(total_gen_disc_loss)

        # If writer exists, write losses
        if exists(self.tb_writer):
            self.tb_writer.add_scalar('loss/rec', self.total_rec_loss, self.steps)
#             self.tb_writer.add_scalar('loss/kl', self.total_kl_loss, self.steps)

        if log:
            self.track(self.total_gen_disc_loss, "G")
            self.track(self.total_l1, "Rec_pips")
            self.track(self.total_l2, "Rec_w")
            self.track(self.total_l3, "Rec_i")
            self.track(self.total_rec_loss, 'Rec')

#         self.track(self.total_kl_loss, 'KL')

        if train:
            self.E_opt.step()

        # calculate moving averages

#         if apply_path_penalty and not np.isnan(avg_pl_length):
#             self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
#             self.track(self.pl_mean, 'PL')

        # save from NaN errors

        if torch.isnan(total_rec_loss):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(num=floor(self.steps / self.evaluate_every))

        self.steps += 1
        self.av = None
                
    @torch.no_grad()
    def evaluate(self, num=0, trunc=1.0):

        self.GAN.eval()
        # ext = self.image_extension  TODO: originally only png if self.transparency was enabled
        ext = "png"
        num_rows = 8 # self.num_image_tiles

        latent_dim = self.encoder.encoder_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        # regular
#         _,image_batch,_ = next(self.loader)
        for _, (_, image_batch) in enumerate(self.test_dataloader):
            break
        image_batch = torch.flatten(image_batch, start_dim=0, end_dim=-4)
        image_batch = image_batch.cuda(self.rank)

        from_encoder_string = "from_encoder"
        with torch.no_grad():
#                 real_classified_logits = self.classifier(image_batch)
            if self.encoder_class == 'GHFeat':
                w = self.encoder(image_batch)
            else:
                w = [(self.encoder(image_batch), num_layers)] 
#             num_rows = len(image_batch)
                     
        # pass images here
        # if w is not None, latents will be ignored.
        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, style=w, noi=n, trunc_psi=self.trunc_psi)
        to_grid = torch.cat((image_batch[:8,:,:,:], generated_images[:8,:,:,:],
                             image_batch[8:16,:,:,:], generated_images[8:16,:,:,:],
                             image_batch[16:24,:,:,:], generated_images[16:24,:,:,:],
                             image_batch[24:,:,:,:], generated_images[24:,:,:,:]))
        torchvision.utils.save_image(to_grid,
                                     str(self.results_dir / f'{str(num)}-{from_encoder_string}.{ext}'),
                                     nrow=num_rows)

        # moving averages
        # generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, style=w, noi=n, trunc_psi=self.trunc_psi)
        # to_grid = torch.cat((image_batch[:8,:,:,:], generated_images[:8,:,:,:],
                             # image_batch[8:16,:,:,:], generated_images[8:16,:,:,:],
                             # image_batch[16:24,:,:,:], generated_images[16:24,:,:,:],
                             # image_batch[24:,:,:,:], generated_images[24:,:,:,:]))
        # torchvision.utils.save_image(to_grid,
                                     # str(self.results_dir / f'{str(num)}-{from_encoder_string}-ema.{ext}'),
                                     # nrow=num_rows)

        # if self.alternating_training:
            
            # if stylevectorizer is trained, then randomly generate some images
            # generated_images = self.generate_from_random(num_rows)
            # torchvision.utils.save_image(generated_images,
                                         # str(self.results_dir / self.name / f'{str(num)}-rand.{ext}'),
                                         # nrow=num_rows)

#         return latents, w

    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi = 0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi = 0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi = trunc_psi)            
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.9, num_image_tiles = 8, demo_batch = None):
        if self.encoder_class != 'GHFeat':        
            w = map(lambda t: (S(t[0]), t[1]), style)
            w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi)
            
            w_styles = styles_def_to_tensor(w_truncated)
        else:
            w_styles = style
            
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
#         if self.conditional and demo_batch is None:
#             return generated_images.clamp_(0., 1.), demo
#         else:
        return generated_images.clamp_(0., 1.)

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid),
            ('Rec', self.total_rec_loss),
            ('Rec_w', self.total_l2),
            ('Rec_i', self.total_l3),
            ('Rec_pips', self.total_l1),
            ('E-G', self.total_gen_disc_loss)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def track(self, value, name):
        
        if name not in self.logger.keys():
            self.logger[name] = []
        else:
            self.logger[name].append(value)

    def model_name(self, model_type, num=None):
        if model_type == 'enc':
            if num is None:
                return self.models_dir / self.encoder_name
            else:
                return self.models_dir / self.encoder_name / f'{model_type}_{num}.pt'
        else:
            if num is None:
                return self.models_dir 
            else:
                return self.models_dir / f'{model_type}_{num}.pt'

    def init_folders(self):
        
        (self.results_dir).mkdir(parents=True, exist_ok=True)
        (self.models_dir).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.encoder_name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir/self.encoder_name), True)
        rmtree(str(self.results_dir), True)
#         rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'load_gan_num': self.load_gan_num,
            'gan_name': self.stylegan_name,
            'encoder_name': self.encoder_name,
            'rec_scaling': self.rec_scaling,
            'logger': self.logger,
            'encoder': self.encoder.state_dict(),
            'discriminator': self.GAN.D.state_dict(),
            'timestamp': datetime.timestamp(datetime.now())
        }

#         if self.GAN.fp16:
#             save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name('enc',num))
        self.write_config()

    def load(self, load_type='model', num = -1):
        self.load_config()
      
        if load_type == 'model':
            del self.GAN
            self.init_GAN()
            pth = Path(self.models_dir).glob('model_*.pt')
        else:
            del self.encoder
            self.init_encoder()
            pth = Path(self.models_dir / self.encoder_name).glob('enc_*.pt')
        
        name = num
        if num == -1:
            file_paths = [p for p in pth]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

#         self.steps = name * self.save_every

        print("Loading", self.model_name(load_type, name))
        load_data = torch.load(self.model_name(load_type, name), map_location="cuda:"+str(self.rank))

        
        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            if load_type == 'model':
                self.load_gan_num = name
                self.GAN.load_state_dict(load_data['GAN'])
            elif load_type == 'enc':
                self.encoder.load_state_dict(load_data['encoder'])
        except Exception as e:
            print('unable to load saved model. please try downgrading the package to the version specified by the saved model')
            raise e
            
        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])
            
