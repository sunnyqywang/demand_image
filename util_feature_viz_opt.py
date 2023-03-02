import numpy as np
import torch
from torch.optim import SGD
from torch.autograd import Variable

import util_model

class GenImagebyOpt():
    def __init__(self, model):
    
        self.model = model
        #self.model.eval()
        self.loss = torch.nn.MSELoss()

    def generate(self, target_var, target_var_val, device, iterations=100, sample=None, learning_rate=4, weight_decay=0.001):
        self.target_var = target_var
        self.target_var_val = torch.Tensor([target_var_val])
        
        loss = []
        
        if sample is None:
            sample = np.uint8(np.random.uniform(0, 255, (1, 3, 224, 224)))
            sample = torch.Tensor(sample)
            sample = sample.to(device)
        else:
            sample = torch.Tensor(sample).to(device)
        sample = Variable(sample, requires_grad=True)
        
        self.target_var_val = self.target_var_val.to(device)
        optimizer = SGD([sample], lr=learning_rate, weight_decay=weight_decay)

        max_value = torch.max(sample).item()
        min_value = torch.min(sample).item()
        sample_clamped = sample

        for i in range(iterations):
            self.model.zero_grad()
            
            # the autoencoder model returns reconstructed image and demographics vector for supervision
            output = self.model(sample_clamped)[1][0,self.target_var]
            
            sample_loss = self.loss(output, self.target_var_val[0])
            loss.append(sample_loss.item())
            
            if i % 20 == 0:
                print("Iteration %d: Target variable value %.4f" % (i, output))
            

            sample_loss.backward(retain_graph=True)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
         
            optimizer.step()
            sample_clamped = torch.clamp(sample, min=min_value, max=max_value)

        return sample_clamped, loss


class DeepDream():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.model_layers = util_model.get_layers(self.model)
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Hook the layers to get result of the convolution
        self.hook_layer()

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.selected_layer.register_forward_hook(hook_function)

    def dream(self, device, iterations=100, sample=None):
        # Process image and return variable
        if sample is None:
            sample = np.uint8(np.random.uniform(0, 255, (1, 3, 224, 224)))
            sample = torch.Tensor(sample)
            sample = sample.to(device)
        else:
            sample = torch.Tensor(sample).to(device)
        sample = Variable(sample, requires_grad=True)

        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas layer layers need less
        optimizer = SGD([sample], lr=12,  weight_decay=1e-4)
        for i in range(iterations):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = sample
            for index, layer in self.model.named_modules():
                print(index, layer)
                # Forward
                # Only need to forward until we the selected layer is reached
                if layer == self.selected_layer:
                    x = layer(x)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            #self.created_image = recreate_image(self.processed_image)
            # Save image every 20 iteration
            # if i % 10 == 0:
                # print(self.created_image.shape)
                # im_path = '../generated/ddream_l' + str(self.selected_layer) + \
                    # '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                # save_image(self.created_image, im_path)
                
        return sample