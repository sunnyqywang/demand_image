import lpips

import torch
from torch import nn, einsum
import torch.nn.functional as F

from util import *
import sys
sys.path.append("models/")
import dcgan
from ghfeat_backup import GHFeat_ResNet_Enc

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
    loss = loss1 + loss2 + loss3
#     loss = loss1 + loss3
    
    return loss    

def classifier_kl_loss(real_classifier_probs, fake_classifier_probs):
    # Convert logits to log_softmax and then KL loss

    # Get probabilities through softmax
#     real_classifier_probabilities = F.log_softmax(real_classifier_logits, dim=1)
#     fake_classifier_probabilities = F.log_softmax(fake_classifier_logits, dim=1)

    real_log = torch.log(real_classifier_probs)
    fake_log = torch.log(fake_classifier_probs)
    
    loss = kl_loss(real_log, fake_log)
    
    return loss

# class StyleVectorizer(nn.Module): ## mapping function z->w
#     def __init__(self, emb, depth, lr_mul = 0.1):
#         super().__init__()
            
#         layers = [EqualLinear(emb, emb, lr_mul), leaky_relu()]
#         for i in range(depth-1):
#             layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

#         self.net = nn.Sequential(*layers)

#     def forward(self, x):   
#         x = F.normalize(x, dim=1)
#         x = self.net(x)

#         return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        
        self.input_channels = input_channels
        self.filters = filters

        self.num_style_coords = self.input_channels + self.filters
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if exists(self.upsample):
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        style_coords = torch.cat([style1, style2], dim=-1)
        
        rgb = self.to_rgb(x, prev_rgb, istyle)
        
        return x, rgb, style_coords
    
class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity=16, transparent=False, attn_layers=[], no_const=False,
                 fmap_max=512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])

        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise, get_style_coords=False):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        if get_style_coords:
            style_coords_list = []

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if exists(attn):
                x = attn(x)

            x, rgb, style_coords = block(x, rgb, style, input_noise)

            if get_style_coords:
                style_coords_list.append(style_coords)

        if get_style_coords:
            style_coords = torch.cat(style_coords_list, dim=1)
            return rgb, style_coords

        else:
            return rgb

    def get_style_coords(self, styles):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)


class DiscriminatorE(nn.Module):
    def __init__(self, image_size, network_capacity=16, fq_layers=[], fq_dict_size=256, attn_layers=[],
                 transparent=False, encoder=False, encoder_dim=512, fmap_max=512):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.encoder_dim = encoder_dim

        if not encoder:
            self.fc = nn.Linear(latent_dim, 1)
        else:
            self.fc = nn.Linear(latent_dim, self.encoder_dim)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            if exists(q_block):
                x, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)

        x = self.fc(x)
        return x.squeeze()


class StylEx(nn.Module):
    def __init__(self, image_size, latent_dim=513, fmap_max=512, style_depth=8, network_capacity=16, transparent=False,
                 fp16=False, cl_reg=False, steps=1, lr=1e-4, ttur_mult=2, fq_layers=[], fq_dict_size=256,
                 attn_layers=[], no_const=False, lr_mlp=0.1, rank=0, encoder_class=None, kl_rec_during_disc=False):
        
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        # DiscriminatorEncoder is default unless specified by encoder class name

        if encoder_class is None:
            self.encoder = DiscriminatorE(image_size, network_capacity, encoder=True, fq_layers=fq_layers,
                                          fq_dict_size=fq_dict_size,
                                          attn_layers=attn_layers, transparent=transparent, fmap_max=fmap_max,
                                          encoder_dim=latent_dim-1)
        elif encoder_class == 'DCGAN':
            model_config = {"base_channels":64, "model_class":"StylEx", "latent_dim":latent_dim-1}
            data_config = {"color_channels":3}
            config = {"model_config": model_config, "data_config": data_config}
            
            self.encoder = dcgan.Discriminator(config)
            self.encoder.encoder_dim = model_config['latent_dim']
        elif encoder_class == 'GHFeat':
            # right now, only works latent dim = 512
            self.encoder = GHFeat_ResNet_Enc(image_size)

        self.S = StyleVectorizer(latent_dim-1, style_depth, condition_dim=0, condition_on_mapper=False, lr_mul=lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent=transparent, attn_layers=attn_layers,
                           no_const=no_const, fmap_max=fmap_max)
        self.D = DiscriminatorE(image_size, network_capacity, fq_layers=fq_layers, fq_dict_size=fq_dict_size,
                                attn_layers=attn_layers, transparent=transparent, fmap_max=fmap_max)

        self.SE = StyleVectorizer(latent_dim-1, style_depth, condition_dim=0, condition_on_mapper=False, lr_mul=lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent=transparent, attn_layers=attn_layers,
                            no_const=no_const)

        self.D_cl = None

        # Is turned off by default
        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam([{'params': generator_params}, {'params': list(self.encoder.parameters()), 'lr': 1e-5}],
                          lr=self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr=self.lr * ttur_mult, betas=(0.5, 0.9))

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda(rank)

        # startup apex mixed precision
        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE, self.encoder), (self.G_opt, self.D_opt) = amp.initialize(
                [self.S, self.G, self.D, self.SE, self.GE, self.encoder], [self.G_opt, self.D_opt], opt_level='O1',
                num_losses=3)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x
