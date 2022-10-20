import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":12})
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

def plot_images(num_row, num_col, img_list, mag=1):
    fig, ax = plt.subplots(num_row, num_col, figsize=(8.5*mag,8.5*mag/num_col*num_row))
    if num_row == 1:
        for j in range(num_col):
            ax[j].imshow(img_list[j])
            ax[j].axis("off")
    else:
        for i in range(num_row):
            for j in range(num_col):
                ax[i,j].imshow(img_list[i][j])
                ax[i,j].axis("off")
    plt.subplots_adjust(
                    wspace=0.05, 
                    hspace=0.05)                
    return fig, ax
    
    
import torch

# def plot_conditional_samples_ssvae(ssvae, visdom_session):
#     """
#     This is a method to do conditional sampling in visdom
#     """
#     vis = visdom_session
#     ys = {}
#     for i in range(10):
#         ys[i] = torch.zeros(1, 10)
#         ys[i][0, i] = 1
#     xs = torch.zeros(1, 784)

#     for i in range(10):
#         images = []
#         for rr in range(100):
#             # get the loc from the model
#             sample_loc_i = ssvae.model(xs, ys[i])
#             img = sample_loc_i[0].view(1, 28, 28).cpu().data.numpy()
#             images.append(img)
#         vis.images(images, 10, 2)


def plot_vae_samples(vae, device, visdom_session):
    vis = visdom_session
    x = torch.zeros([1,3,224,224])
    x = x.to(device)
    for i in range(10):
        images = []
        for rr in range(100):
            # get loc from the model
            sample_loc_i = vae.model(x)
            img = sample_loc_i[0].view(3, 224, 224).cpu().data.numpy()
            images.append(img)
        vis.images(images, 10, 2)

        
def plot_latent_class(z_loc, classes, class_type='autoshare'):
    
#     z_loc, classes = latent_space(vae, test_loader, device, demo_cs, demo_np, label_index, num_clusters, class_type)

    model_tsne = TSNE(n_components=2, random_state=42)
    z_states = z_loc
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes
    fig, ax = plt.subplots(1, figsize=(4,4))
    for ic in range(max(classes)+1):
        ind_class = classes == ic
        color = plt.cm.Set1(ic)
        ax.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=5, color=color, label='Class '+str(ic), alpha = 0.5)
        ax.set_title("Latent Variable T-SNE (" + class_type + ")")
    ax.legend()
    return fig

def plot_latent_density(z_loc, ax):
#     z_loc, classes = latent_space(vae, test_loader, device, demo_cs, demo_np, label_index=None)
    model_tsne = TSNE(n_components=2, random_state=42)
    z_states = z_loc
    z_embed = model_tsne.fit_transform(z_states)
    z_embed = pd.DataFrame(z_embed, columns=['x1','x2'])
    ax = sns.kdeplot(data = z_embed, x='x1', y='x2', ax = ax)
    
    return ax