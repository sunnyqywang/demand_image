import cv2
import numpy as np
import torch
import torchvision

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
    
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img
    
def single_image_to_tensor_input(img_dir,crop_size):

    sample = image_transform(cv2.imread(img_dir), crop_size=crop_size)
    sample = sample.reshape((1,)+sample.shape)
    
    return sample

def image_transform(img, crop_size, recalculate_normalize=False):
    if recalculate_normalize:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])       
        trainset = ImageDataset(image_dir, data_dir, train=False, transform=transform)
        all_images = trainset[0][1].reshape(3, -1)
        for i in range(1,len(trainset)):
            all_images = torch.cat((all_images, trainset[i][1].reshape(3, -1)), dim=1)
        mean = torch.mean(all_images, axis=1)
        std = torch.std(all_images, axis=1)

        print("Satellite Mean: ", mean)
        print("Satellite Std:", std)
    else:
        mean = [0.3733, 0.3991, 0.3711]
        std = [0.2173, 0.2055, 0.2143]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.Normalize(mean, std)
    ])

    img_transformed = transform(img)
    
    return img_transformed
    
def inverse_transform(img, mean=[], std=[], grayscale=False):
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-m/x for m,x in zip(mean, std)],
        std=[1/x for x in std]
    )
    
    img = torch.Tensor(img)
    img_orig = inv_normalize(img)
    if grayscale:
        pass
        # implement transformation to grayscale
    
    return img_orig
    
