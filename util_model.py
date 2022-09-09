import sys
sys.path.append("models/")
import importlib
import pandas as pd
import torch


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
