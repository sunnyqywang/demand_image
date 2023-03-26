import os
import sys
import math
import fire
import json

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
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

from einops import rearrange, repeat
from kornia.filters import filter2d

import torchvision
from torchvision import transforms
# from version import __version__
# from diff_augment import DiffAugment

from vector_quantize_pytorch import VectorQuantize

from PIL import Image
from pathlib import Path

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import aim

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# Classifier
import sys
sys.path.append("models/")
from mobilenet import MobileNetV2
# from resnet_classifier import ResNet

# Encoders for debugging or additional testing
# import debug_encoders
from util import *
from stylex import *
# constants

NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']


class Trainer():
    def __init__(
            self,
            name='default',
            results_dir='results',
            models_dir='models',
            base_dir='./',
            image_size=128,
            network_capacity=16,
            fmap_max=512,
            transparent=False,
            batch_size=8,
            mixed_prob=0.9,
            gradient_accumulate_every=6,
            lr=2e-4,
            lr_mlp=0.1,
            ttur_mult=2,
            rel_disc_loss=False,
            num_workers=None,
            save_every=1000,
            evaluate_every=1000,
            num_image_tiles=8,
            trunc_psi=0.6,
            fp16=False,
            cl_reg=False,
            no_pl_reg=False,
            fq_layers=[],
            fq_dict_size=256,
            attn_layers=[],
            no_const=False,
            aug_prob=0.,
            aug_types=['translation', 'cutout'],
            top_k_training=False,
            generator_top_k_gamma=0.99,
            generator_top_k_frac=0.5,
            dual_contrast_loss=False,
            dataset_aug_prob=0.,
            calculate_fid_every=None,
            calculate_fid_num_images=12800,
            clear_fid_cache=False,
            is_ddp=False,
            rank=0,
            world_size=1,
            log=False,
            kl_scaling=1,
            rec_scaling=10, 
            # below are stylex params
            classifier_path="classifier.pt",  # path to classifier
            num_classes=2,  # num_classes
            encoder_class=None,  # encoder class None defaults to discriminator encoder
            alternating_training=True,
            sample_from_encoder=False,
            tensorboard_dir=None,
            classifier_name=None, # classifier model class
            *args,
            **kwargs):
        
        self.model_params = [args, kwargs]
        self.StylEx = None

        self.kl_scaling = kl_scaling
        self.rec_scaling = rec_scaling

        self.alternating_training = alternating_training

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir / 'sGAN2'
        self.models_dir = base_dir / models_dir / 'sGAN2'
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.models_dir / name / '.config.json'
        print("Results directory:", self.results_dir / self.name)
        print("Model directory:", self.models_dir / self.name)

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
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
        self.total_rec_loss = 0
        self.total_kl_loss = 0
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
        self.is_main = True
        self.rank = rank
        self.world_size = world_size
        self.sample_from_encoder = sample_from_encoder

        self.logger = aim.Session(experiment=name) if log else None

        # Load classifier
        self.num_classes = num_classes
        self.classifier = None
        # resnet classifier not implemented yet
#         if classifier_name.lower() == "resnet":
#             self.classifier = ResNet(classifier_path, cuda_rank=rank, output_size=self.num_classes,
#                                      image_size=image_size)
#         else:
        print("Loading classifier from: ", classifier_path)
        self.classifier = MobileNetV2({'condition_dim': num_classes-1, 'image_size': image_size})
        saved = torch.load(classifier_path, map_location=torch.device("cuda:"+str(self.rank)))
        self.classifier.load_state_dict(saved['model_state_dict'])
        self.classifier = self.classifier.cuda(self.rank)
        
        # Load tensorboard, create writer
        self.tb_writer = None
        if exists(tensorboard_dir):
            self.tb_writer = SummaryWriter(os.path.join(tensorboard_dir, name))
            
        self.lpips_loss = lpips.LPIPS(net="alex").cuda(self.rank) # image should be RGB, IMPORTANT: normalized to [-1,1]

        self.encoder_class = encoder_class
        
    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}

    def init_StylEx(self):
        args, kwargs = self.model_params
        self.StylEx = StylEx(image_size=self.image_size, 
                             lr=self.lr, lr_mlp=self.lr_mlp, ttur_mult=self.ttur_mult, 
                             network_capacity=self.network_capacity, fmap_max=self.fmap_max,
                             transparent=self.transparent, fq_layers=self.fq_layers, fq_dict_size=self.fq_dict_size,
                             attn_layers=self.attn_layers, fp16=self.fp16, cl_reg=self.cl_reg, no_const=self.no_const,
                             rank=self.rank, encoder_class=self.encoder_class, *args, **kwargs)

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank]}
            self.S_ddp = DDP(self.StylEx.S, **ddp_kwargs)
            self.G_ddp = DDP(self.StylEx.G, **ddp_kwargs)
            self.D_ddp = DDP(self.StylEx.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.StylEx.D_aug, **ddp_kwargs)

        if exists(self.logger):
            self.logger.set_params(self.hparams)
            
            
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
        del self.StylEx
        self.init_StylEx()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp,
                'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size,
                'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, folder, batch_size=None):
        
#         self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        transform = transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Resize(self.image_size)])
#             torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    
        if batch_size is None:
            batch_size = self.batch_size
            
        self.dataset = ImageDataset(folder, data_dir, train=True, data_version='1571', transform=transform, sampling='clustered', image_type='png', augment=None, demo=True)
        num_workers = num_workers = default(self.num_workers, NUM_CORES if not self.is_ddp else 0)
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)
        
        self.set_test_data_src(folder, 32)        
        
        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def set_test_data_src(self, folder, batch_size):
        
        transform = transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Resize(self.image_size)])
        
        self.test_dataset = ImageDataset(folder, data_dir, train=False, data_version='1571', transform=transform, sampling='clustered', image_type='png', augment=None, demo=True)
        num_workers = num_workers = default(self.num_workers, NUM_CORES if not self.is_ddp else 0)
        self.test_dataloader = data.DataLoader(self.test_dataset, num_workers = num_workers, batch_size = batch_size, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        
        
    def train_encoder_only(self):
        assert exists(self.StylEx), 'Must load trained StylEx G, D, and S to train encoder only'
        
        self.StylEx.encoder.train()
        
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_rec_loss = torch.tensor(0.).cuda(self.rank)
        batch_size = math.ceil(self.batch_size / self.world_size)
        
        
        image_size = self.StylEx.G.image_size
        latent_dim = self.StylEx.G.latent_dim
        num_layers = self.StylEx.G.num_layers

        aug_prob = self.aug_prob
        aug_types = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}
        
        G = self.StylEx.G if not self.is_ddp else self.G_ddp
        D = self.StylEx.D if not self.is_ddp else self.D_ddp
        D_aug = self.StylEx.D_aug if not self.is_ddp else self.D_aug_ddp

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        backwards = partial(loss_backwards, self.fp16)
        
        D_loss_fn = hinge_loss
        self.StylEx.D_opt.zero_grad()
        
        E_opt = Adam(self.StylEx.encoder.parameters(), lr=self.lr, betas=(0.5, 0.9))
        E_opt.zero_grad()
        
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug]):
            _,image_batch,_ = next(self.loader)
            image_batch = image_batch.cuda(self.rank)
            image_batch.requires_grad_()
            
#             discriminator_batch = next(self.loader).cuda(self.rank)
#             discriminator_batch.requires_grad_()

            
            encoder_output = self.StylEx.encoder(image_batch)
            real_classified_logits = self.classifier(image_batch)
            style = [(torch.cat((encoder_output, real_classified_logits), dim=1),
                      self.StylEx.G.num_layers)]  # Has to be bracketed because expects a noise mix
            noise = image_noise(batch_size, image_size, device=self.rank)
      
            w_styles = styles_def_to_tensor(style)
            
            encoder_input = False
            
            generated_images = G(w_styles, noise)
            fake_output = D_aug(generated_images.clone().detach(), detach=True, **aug_kwargs)

            real_output = D_aug(image_batch, **aug_kwargs)

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
            backwards(disc_loss, self.StylEx.D_opt, loss_id=1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')

        self.StylEx.D_opt.step()
        
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[]):
            _,image_batch,_ = next(self.loader)
            image_batch = image_batch.cuda(self.rank)
            image_batch.requires_grad_()
            
            encoder_output = self.StylEx.encoder(image_batch)
            real_classified_logits = self.classifier(image_batch)

            style = [(torch.cat((encoder_output, real_classified_logits), dim=1), self.StylEx.G.num_layers)]
            noise = image_noise(batch_size, image_size, device=self.rank)

            w_styles = styles_def_to_tensor(style)
            
            generated_images = G(w_styles, noise)

#             fake_output = D_aug(generated_images, **aug_kwargs)
#             fake_output_loss = fake_output

            # Our losses
            if not self.alternating_training or encoder_input:
                # multiply losses by 2 since they are only calculated every other iteration if using alternating training
                rec_loss = 2 * self.rec_scaling * reconstruction_loss(image_batch, generated_images,
                                                                      self.StylEx.encoder(generated_images),
                                                                      encoder_output, self.lpips_loss) / self.gradient_accumulate_every
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


            backwards(rec_loss, E_opt, loss_id=2, retain_graph=True)
#             backwards(kl_loss, self.StylEx.G_opt, loss_id=3)

            total_rec_loss += rec_loss.detach().item()
#             total_kl_loss += kl_loss.detach().item()

            self.total_rec_loss = float(total_rec_loss)
#             self.total_kl_loss = float(total_kl_loss)

        # If writer exists, write losses
        if exists(self.tb_writer):
            self.tb_writer.add_scalar('loss/rec', self.total_rec_loss, self.steps)
#             self.tb_writer.add_scalar('loss/kl', self.total_kl_loss, self.steps)

        self.track(self.total_rec_loss, 'Rec')
#         self.track(self.total_kl_loss, 'KL')

        E_opt.step()

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
                self.evaluate(encoder_input=self.sample_from_encoder, num=floor(self.steps / self.evaluate_every), encoder_training=True)

        self.steps += 1
        self.av = None
                
        
    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.StylEx):
            self.init_StylEx()

        self.StylEx.encoder.train()
        self.StylEx.train()
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_loss = torch.tensor(0.).cuda(self.rank)
        total_rec_loss = torch.tensor(0.).cuda(self.rank)
        total_kl_loss = torch.tensor(0.).cuda(self.rank)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.StylEx.G.image_size
        latent_dim = self.StylEx.G.latent_dim
        num_layers = self.StylEx.G.num_layers

        aug_prob = self.aug_prob
        aug_types = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        S = self.StylEx.S if not self.is_ddp else self.S_ddp
        G = self.StylEx.G if not self.is_ddp else self.G_ddp
        D = self.StylEx.D if not self.is_ddp else self.D_ddp
        D_aug = self.StylEx.D_aug if not self.is_ddp else self.D_aug_ddp

        backwards = partial(loss_backwards, self.fp16)

        # setup losses

        if not self.dual_contrast_loss:
            D_loss_fn = hinge_loss
            G_loss_fn = gen_hinge_loss
            G_requires_reals = False
        else:
            D_loss_fn = dual_contrastive_loss
            G_loss_fn = dual_contrastive_loss
            G_requires_reals = True

        # train discriminator

        avg_pl_length = self.pl_mean
        self.StylEx.D_opt.zero_grad()

        if self.alternating_training:
            encoder_input = False

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, S, G]):
            _,image_batch,_ = next(self.loader)
            image_batch = image_batch.cuda(self.rank)
            image_batch.requires_grad_()
            
            if not self.alternating_training or encoder_input:
                
                encoder_output = self.StylEx.encoder(image_batch)
                real_classified_logits = self.classifier(image_batch)
                noise = image_noise(batch_size, image_size, device=self.rank)
                if self.encoder_class != 'GHFeat':
                    style = [(torch.cat((encoder_output, real_classified_logits), dim=1),
                              self.StylEx.G.num_layers)]  # Has to be bracketed because expects a noise mix

                    w_styles = styles_def_to_tensor(style)
                else:
#                     print(encoder_output.shape)
#                     print(real_classified_logits.shape)
#                     print(real_classified_logits[:,None,:].expand(-1,self.StylEx.G.num_layers,-1).shape)
                    w_styles = torch.cat((encoder_output, real_classified_logits[:,None,:].expand(-1,self.StylEx.G.num_layers,-1)), dim=2)
                    
                encoder_input = False
                
            else:
                get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
                style = get_latents_fn(batch_size, num_layers, self.StylEx.encoder.encoder_dim, device=self.rank)
                noise = image_noise(batch_size, image_size, device=self.rank)

                w_space = latent_to_w(S, style)
                c = torch.rand(self.batch_size,1).cuda(self.rank)
                w_space = [(torch.cat((w, c), dim=1), l) for w, l in w_space]
                w_styles = styles_def_to_tensor(w_space)

                if self.alternating_training:
                    encoder_input = True

            generated_images = G(w_styles, noise)
            fake_output = D_aug(generated_images.clone().detach(), detach=True, **aug_kwargs)

            real_output = D_aug(image_batch, **aug_kwargs)

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
            backwards(disc_loss, self.StylEx.D_opt, loss_id=1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')

        self.StylEx.D_opt.step()

        # train generator

        if self.alternating_training:
            encoder_input = False

        self.StylEx.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[S, G, D_aug]):
            
#             _,image_batch,_ = next(self.loader)
#             image_batch = image_batch.cuda(self.rank)
#             image_batch.requires_grad_()

            if not self.alternating_training or encoder_input:
                encoder_output = self.StylEx.encoder(image_batch)
                real_classified_logits = self.classifier(image_batch)
                noise = image_noise(batch_size, image_size, device=self.rank)
                if self.encoder_class != 'GHFeat':
                    style = [(torch.cat((encoder_output, real_classified_logits), dim=1), self.StylEx.G.num_layers)]
                    w_styles = styles_def_to_tensor(style)
                else:
                    w_styles = torch.cat((encoder_output, real_classified_logits[:,None,:].expand(-1,self.StylEx.G.num_layers,-1)), dim=2)
            else:
                style = get_latents_fn(batch_size, num_layers, self.StylEx.encoder.encoder_dim, device=self.rank)
                noise = image_noise(batch_size, image_size, device=self.rank)

                w_space = latent_to_w(S, style)
                c = torch.rand(self.batch_size,1).cuda(self.rank)
                w_space = [(torch.cat((w, c), dim=1), l) for w, l in w_space]
                w_styles = styles_def_to_tensor(w_space) 
                
            generated_images = G(w_styles, noise)
            gen_image_classified_logits = self.classifier(generated_images)

            fake_output = D_aug(generated_images, **aug_kwargs)
            fake_output_loss = fake_output

            real_output = None
            if G_requires_reals:
                image_batch = next(self.loader).cuda(self.rank)
                real_output, _ = D_aug(image_batch, detach=True, **aug_kwargs)
                real_output = real_output.detach()

            if self.top_k_training:
                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

            # Our losses
            if not self.alternating_training or encoder_input:
                # multiply losses by 2 since they are only calculated every other iteration if using alternating training
                rec_loss = 2 * self.rec_scaling * reconstruction_loss(image_batch, generated_images,
                                                                      self.StylEx.encoder(generated_images),
                                                                      encoder_output, self.lpips_loss) / self.gradient_accumulate_every
#                 rec_loss.cuda(self.rank)
                kl_loss = 2 * self.kl_scaling * classifier_kl_loss(real_classified_logits,
                                                                   gen_image_classified_logits) / self.gradient_accumulate_every
        
            # Original loss
            loss = G_loss_fn(fake_output_loss, real_output)
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)

            if not self.alternating_training or encoder_input:

                backwards(gen_loss, self.StylEx.G_opt, loss_id=2, retain_graph=True)
                backwards(rec_loss, self.StylEx.G_opt, loss_id=3, retain_graph=True)
                backwards(kl_loss, self.StylEx.G_opt, loss_id=4)

                total_gen_loss += loss.detach().item() / self.gradient_accumulate_every
                total_rec_loss += rec_loss.detach().item()
                total_kl_loss += kl_loss.detach().item()

                self.g_loss = float(total_gen_loss)
                self.total_rec_loss = float(total_rec_loss)
                self.total_kl_loss = float(total_kl_loss)
            else:
                backwards(gen_loss, self.StylEx.G_opt, loss_id=2)

                total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

                self.g_loss = float(total_gen_loss)

            encoder_input = not encoder_input

        # If writer exists, write losses
        if exists(self.tb_writer):
            self.tb_writer.add_scalar('loss/G', self.g_loss, self.steps)
            self.tb_writer.add_scalar('loss/D', self.d_loss, self.steps)
            self.tb_writer.add_scalar('loss/rec', self.total_rec_loss, self.steps)
            self.tb_writer.add_scalar('loss/kl', self.total_kl_loss, self.steps)

        self.track(self.g_loss, 'G')
        self.track(self.total_rec_loss, 'Rec')
        self.track(self.total_kl_loss, 'KL')

        self.StylEx.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.StylEx.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.StylEx.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(encoder_input=self.sample_from_encoder, num=floor(self.steps / self.evaluate_every))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

        self.steps += 1
        self.av = None
    
    @torch.no_grad()
    def evaluate(self, encoder_input=False, num=0, trunc=1.0, encoder_training=False):
        self.StylEx.eval()
        # ext = self.image_extension  TODO: originally only png if self.transparency was enabled
        ext = "png"
        num_rows = 8 # self.num_image_tiles

        latent_dim = self.StylEx.encoder.encoder_dim
        image_size = self.StylEx.G.image_size
        num_layers = self.StylEx.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        # regular
        from_encoder_string = ""
#         _,image_batch,_ = next(self.loader)
        for _, (_, image_batch, _) in enumerate(self.test_dataloader):
            break
        image_batch = image_batch.cuda(self.rank)

        if encoder_input:
            from_encoder_string = "from_encoder"
            with torch.no_grad():
                real_classified_logits = self.classifier(image_batch)
                if self.encoder_class == 'GHFeat':
                    w = torch.cat((self.StylEx.encoder(image_batch), real_classified_logits[:,None,:].expand(-1,num_layers,-1)), dim=2)
                else:
                    w = [(self.StylEx.encoder(image_batch), num_layers)]
#             num_rows = len(image_batch)
        else:
            w = None
            real_classified_logits = None
            
        if encoder_training:
            from_encoder_string = "train_encoder"
            
        # pass images here
        # if w is not None, latents will be ignored.
        generated_images = self.generate_truncated(self.StylEx.S, self.StylEx.G, latents, n, w=w, c=real_classified_logits,
                                                   trunc_psi=self.trunc_psi)
        to_grid = torch.cat((image_batch[:8,:,:,:], generated_images[:8,:,:,:],
                             image_batch[8:16,:,:,:], generated_images[8:16,:,:,:],
                             image_batch[16:24,:,:,:], generated_images[16:24,:,:,:],
                             image_batch[24:,:,:,:], generated_images[24:,:,:,:]))
        torchvision.utils.save_image(to_grid,
                                     str(self.results_dir / self.name / f'{str(num)}-{from_encoder_string}.{ext}'),
                                     nrow=num_rows)

        # moving averages
        generated_images = self.generate_truncated(self.StylEx.SE, self.StylEx.GE, latents, n, w=w, c=real_classified_logits,
                                                   trunc_psi=self.trunc_psi)
        to_grid = torch.cat((image_batch[:8,:,:,:], generated_images[:8,:,:,:],
                             image_batch[8:16,:,:,:], generated_images[8:16,:,:,:],
                             image_batch[16:24,:,:,:], generated_images[16:24,:,:,:],
                             image_batch[24:,:,:,:], generated_images[24:,:,:,:]))
        torchvision.utils.save_image(to_grid,
                                     str(self.results_dir / self.name / f'{str(num)}-{from_encoder_string}-ema.{ext}'),
                                     nrow=num_rows)

        if self.alternating_training:
            
            # if stylevectorizer is trained, then randomly generate some images
            generated_images = self.generate_from_random(num_rows)
            torchvision.utils.save_image(generated_images,
                                         str(self.results_dir / self.name / f'{str(num)}-rand.{ext}'),
                                         nrow=num_rows)

    @torch.no_grad()
    def generate_from_random(self, num_rows):
        latent_dim = self.StylEx.encoder.encoder_dim
        image_size = self.StylEx.G.image_size
        num_layers = self.StylEx.G.num_layers

        # latents and noise
        latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        generated_image = self.generate_truncated(self.StylEx.SE, self.StylEx.GE, latents, n, w=None, c=None, 
                                                   trunc_psi=self.trunc_psi)
        
        return generated_image
    
    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi=0.75):
        S = self.StylEx.S
        batch_size = self.batch_size
        latent_dim = self.StylEx.encoder.encoder_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis=0)
            self.av = np.expand_dims(self.av, axis=0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi=0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi=trunc_psi)
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, w=None, c=None, trunc_psi=0.75, num_image_tiles=8):
        if w is None:
            w = map(lambda t: (S(t[0]), t[1]), style)
        if c is None:
            c = torch.rand(noi.shape[0], self.num_classes-1).cuda(self.rank)

        if self.encoder_class != 'GHFeat':
            w_truncated = self.truncate_style_defs(w, trunc_psi=trunc_psi)
            w_truncated = [(torch.cat((w, c), dim=1), l) for w, l in w_truncated]
            w_styles = styles_def_to_tensor(w_truncated)
        else:
            w_styles = w
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num=0, num_image_tiles=8, trunc=1.0, num_steps=100, save_frames=False):
        self.StylEx.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.StylEx.G.latent_dim
        image_size = self.StylEx.G.image_size
        num_layers = self.StylEx.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.StylEx.SE, self.StylEx.GE, latents, n,
                                                       trunc_psi=self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow=num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())

            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)

            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:],
                       duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

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
            ('KL', self.total_kl_loss)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.track(value, name=name)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'StylEx': self.StylEx.state_dict(),
        }

        if self.StylEx.fp16:
            save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name), map_location="cuda:"+str(self.rank))
        
        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            self.StylEx.load_state_dict(load_data['StylEx'])
        except Exception as e:
            print(
                'unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
        if self.StylEx.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])
            
    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # remove any existing files used for fid calculation and recreate directories

        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    filename = str(k + batch_num * self.batch_size)
                    torchvision.utils.save_image(image, str(real_path / f'{filename}.png'))

        # generate a bunch of fake images in results / name / fid_fake

        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.StylEx.eval()
        ext = self.image_extension

        latent_dim = self.StylEx.G.latent_dim
        image_size = self.StylEx.G.image_size
        num_layers = self.StylEx.G.num_layers

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(self.batch_size, image_size, device=self.rank)

            # moving averages
            generated_images = self.generate_truncated(self.StylEx.SE, self.StylEx.GE, latents, noise,
                                                       trunc_psi=self.trunc_psi)

            for j, image in enumerate(generated_images.unbind(0)):
                torchvision.utils.save_image(image,
                                             str(fake_path / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, noise.device, 2048)
