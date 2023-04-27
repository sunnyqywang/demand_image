import sys
sys.path.append("models/stylegan2/")
sys.path.append(".")
from sgan2_enc_trainer import Trainer

import os
import random
from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from functools import wraps
from util import NanException

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np

from setup import image_dir, out_dir, model_dir

def encoder_training(rank, world_size, model_args, data, gan_load_from, enc_load_from, new, num_train_steps, seed):
#     is_main = rank == 0
    is_main = True
    is_ddp = world_size > 1

    if is_ddp:
        set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(
        is_ddp = is_ddp,
        rank = rank,
        world_size = world_size
    )

    model = Trainer(**model_args)
    
    model.load('model', gan_load_from)
    if not new:
        model.load('encoder', enc_load_from)
    else:
        model.clear()

    model.set_data_src(data)
    model.set_test_data_src(data, 8)

    # progress_bar = tqdm(initial = model.steps, total = num_train_steps, mininterval=10., desc=f'<{data}>', position=0, leave=True)
    while model.steps < num_train_steps:
        # retry_call(model.train_encoder_only, tries=3, exceptions=NanException)
        # progress_bar.n = model.steps
        # progress_bar.refresh()
        model.train_encoder_only()
        if is_main and model.steps % 500 == 0:
            model.print_log()
            
    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()
        
if __name__ == '__main__':
    data = image_dir
    results_dir = out_dir
    models_dir = model_dir
    stylegan_name = '2303-2'
    encoder_name = '0426-50'

    # GAN
    gan_load_from = 100
    # encoder
    new = True
    enc_load_from = -1

    image_size = 64
    network_capacity = 16
    fmap_max = 512
    transparent = False
    batch_size = 4
    gradient_accumulate_every = 6
    num_train_steps = 100000
    learning_rate = 1e-5
    lr_mlp = 0.1
    ttur_mult = 1.5
    rel_disc_loss = False
    num_workers =  None
    save_every = 1000
    evaluate_every = 1000
    generate = False
    num_generate = 1
    generate_interpolation = False
    interpolation_num_steps = 100
    save_frames = False
    num_image_tiles = 8
    trunc_psi = 0.75
    mixed_prob = 0.9
    fp16 = False
    no_pl_reg = False
    cl_reg = False
    fq_layers = []
    fq_dict_size = 256
    attn_layers = []
    no_const = False
    aug_prob = 0.
    aug_types = ['translation', 'cutout']
    top_k_training = False
    generator_top_k_gamma = 0.99
    generator_top_k_frac = 0.5
    dual_contrast_loss = False
    dataset_aug_prob = 0.
    multi_gpus = True
    calculate_fid_every = None
    calculate_fid_num_images = 12800
    clear_fid_cache = False
    seed = 42

    # A global scale to the custom losses
    rec_w_scaling=1
    rec_scaling=50

    # If unspecified, use the Discriminator as an encoder (like the authors did).
    # This is the way to go if we want to be close to the original paper.
    # Check out debug_encoders.py for the names of classes if you still want
    # to use a different encoder.
    encoder_class='GHFeat'

    kl_rec_during_disc=False

    # This is for making the image results be results of the
    # image -> encoder -> generator pipeline
    # Set False if training a standard GAN or if you want to see
    # examples from a noise vector.
    sample_from_encoder=True

    # Alternatively trains the model with the StylEx loss
    # and the regular StyleGAN loss. If False just trains
    # using the encoder.
    alternating_training=False

    tensorboard_dir=None  # Put to None for not logging
    rank = 0


    model_args = dict(
            stylegan_name=stylegan_name,
            encoder_name=encoder_name,
            results_dir=results_dir,
            models_dir=models_dir,
            batch_size=batch_size,
            gradient_accumulate_every=gradient_accumulate_every,
            image_size=image_size,
            network_capacity=network_capacity,
            fmap_max=fmap_max,
            transparent=transparent,
            lr=learning_rate,
            lr_mlp=lr_mlp,
            ttur_mult=ttur_mult,
            rel_disc_loss=rel_disc_loss,
            num_workers=num_workers,
            save_every=save_every,
            evaluate_every=evaluate_every,
            num_image_tiles=num_image_tiles,
            trunc_psi=trunc_psi,
            fp16=fp16,
            no_pl_reg=no_pl_reg,
            cl_reg=cl_reg,
            fq_layers=fq_layers,
            fq_dict_size=fq_dict_size,
            attn_layers=attn_layers,
            no_const=no_const,
            aug_prob=aug_prob,
            aug_types=aug_types,
            top_k_training=top_k_training,
            generator_top_k_gamma=generator_top_k_gamma,
            generator_top_k_frac=generator_top_k_frac,
            dual_contrast_loss=dual_contrast_loss,
            dataset_aug_prob=dataset_aug_prob,
            calculate_fid_every=calculate_fid_every,
            calculate_fid_num_images=calculate_fid_num_images,
            clear_fid_cache=clear_fid_cache,
            mixed_prob=mixed_prob,
            rec_w_scaling=rec_w_scaling,
            rec_scaling=rec_scaling,
    #         classifier_path=classifier_path,
    #         num_classes=num_classes,
            encoder_class=encoder_class,
            sample_from_encoder=sample_from_encoder,
            alternating_training=alternating_training,
            kl_rec_during_disc=kl_rec_during_disc,
            tensorboard_dir=tensorboard_dir,
    #         classifier_name=classifier_name,
            rank=rank)
    world_size = 1
    
    encoder_training(rank, world_size, model_args, data, gan_load_from, enc_load_from, new, num_train_steps, seed)
    
    
