import pandas as pd
import torch

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
    
def my_loss(out_image, out_demo, data, census_data, factor=20):
    reconstruct_loss = torch.mean((out_image - data)**2)
    regression_loss = torch.mean((out_demo - census_data)**2)
    # print(reconstruct_loss, regression_loss)
    return reconstruct_loss + regression_loss * factor

