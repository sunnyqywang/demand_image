import torch
import os
import h5py

from util import *

from PIL import ImageDraw
from PIL import ImageFont

def sindex_to_block_idx_and_index(generator, sindex):
    tmp_idx = sindex

    block_idx = None
    idx = None

    for idx, block in enumerate(generator.blocks):
        
        if tmp_idx < block.num_style_coords:
            block_idx = idx
            idx = tmp_idx
            break
        else:
            tmp_idx = tmp_idx - block.num_style_coords

    return block_idx, idx

def get_min_max_style_vectors(style_coordinates):
    
    # element-wise minimums and maximums
    minimums = None
    maximums = None

    for style_coords in style_coordinates:
        if minimums is None or maximums is None:
            minimums = style_coords
            maximums = style_coords
        else:
            minimums = torch.minimum(minimums, style_coords)
            maximums = torch.maximum(maximums, style_coords)
    
    if minimums == None:
        raise ValueError('No images pass the threshold check')

    return minimums, maximums

def discriminator_filter(discriminator, generated_image, threshold, probabilities=None):
    
    # probabilities always seem to be None
    # dont really know what probabilities do
    if probabilities is not None:
        output_generated = discriminator(generated_image, probabilities=probabilities)
    else:
        output_generated = discriminator(generated_image)
    
    if threshold == None:
        return output_generated
    if output_generated < threshold:
        return (False, output_generated)
    else:
        return (True, output_generated)

def attfind_extraction(dataloader,
                       num_images,
                       results_folder,
                       stylex,
                       classifier,
                       dataset_name,
                       noise,
                       num_style_coords,
                       shift_size,
                       discriminator_threshold,
                       image_size=64,
                       batch_size=1,
                       cuda_rank = 0,
                       use_discriminator=False):
    
    if batch_size != 1:
        raise ValueError('Please use a batch_size equal to 1')

    with torch.no_grad():
        
        style_change_effects = torch.zeros((num_images, 2, num_style_coords, 2)).cuda(cuda_rank)
        image_latents = torch.zeros((num_images, 513)).cuda(cuda_rank)
        generated_image_classifications = torch.zeros((num_images, 1)).cuda(cuda_rank)
        style_coordinates = torch.zeros((num_images, num_style_coords)).cuda(cuda_rank)
        original_images = torch.zeros((num_images, 3, image_size, image_size)).cuda(cuda_rank)
        discriminator_results = torch.zeros((num_images, 1)).cuda(cuda_rank)
        
        # Generate the images we want to use (use discriminator to filter if 'use_discriminator' is True
        images_found = 0

#         dataloader = iter(dataloader)
#         for _,batch,_ in dataloader:

        while (images_found < num_images):
    
            print(f"\rFound {images_found}/{num_images} images".format(images_found), end="")
#             if images_found >= num_images:
#                 break

#             batch = batch.cuda(cuda_rank)
#             # encode real images
#             encoder_output = stylex.encoder(batch).unsqueeze(0)
#             # classification results
#             real_classified_logits = classifier(batch)


            # generate according to encodings
#             concat_w_tensor = torch.cat((encoder_output, real_classified_logits), dim=1)
#             latent_w = [(concat_w_tensor, stylex.G.num_layers)]

            # Generate from random
            latents = torch.randn(1, stylex.encoder.encoder_dim).cuda(cuda_rank) 
            c = torch.rand(1, 1).cuda(cuda_rank)
            concat_w_tensor = torch.cat((latents, c), dim=1)
            latent_w = [(concat_w_tensor, stylex.G.num_layers)] 
            
            w_latent_tensor = styles_def_to_tensor(latent_w)
            # get generated image and style coordinates
            generated_image, style_coords = stylex.G(w_latent_tensor, noise, get_style_coords=True)

            # Filter images deemed fake by the discriminator
            skip, discriminator_output = discriminator_filter(stylex.D, generated_image, discriminator_threshold)
            
            if use_discriminator and skip:
                # if image is fake, then regenerate
                continue
            else:
#                 original_images[images_found] = batch
                original_images[images_found] = generated_image
                image_latents[images_found] = concat_w_tensor
                style_coordinates[images_found] = style_coords
                discriminator_results[images_found] = discriminator_output
                generated_image_classifications[images_found] = classifier(generated_image)

                images_found += 1

        if images_found < num_images:
            print()
            print("Not enough images.")
            return
        
        print()
        print("Retrieving min and max") # element-wise
        minima, maxima = get_min_max_style_vectors(style_coordinates)
        
        print("Exploring StyleSpace attributes")
        
        # for filtered_image in filtered_images:  # TODO: Change this to a for loop
        image_index = 0
        
        for image_index in range(images_found):
            
            style_coords = style_coordinates[image_index]
            image_generated_logits = generated_image_classifications[image_index]

            concat_w_tensor = image_latents[image_index].unsqueeze(0)
            latent_w = [(concat_w_tensor, stylex.G.num_layers)]
            w_latent_tensor = styles_def_to_tensor(latent_w)

            # shape: (1, num directions, num style coords, num classes)
            style_change_effect = torch.Tensor(1, 2, num_style_coords, 1).cuda(cuda_rank)

            for sindex in tqdm(range(num_style_coords)):

                # sindex is the style coordinate in the flattened vector
                # get block and weight index in the hierarchical structure
                block_idx, weight_idx = sindex_to_block_idx_and_index(stylex.G, sindex)
                block = stylex.G.blocks[block_idx]

                # get one-hot representation of the style coordinate (so we can modify it)
                current_style_layer = None
                one_hot = None

                if weight_idx < block.input_channels:
                    current_style_layer = block.to_style1
                    one_hot = torch.zeros((1, block.input_channels)).cuda(cuda_rank)
                else:
                    weight_idx -= block.input_channels
                    current_style_layer = block.to_style2
                    one_hot = torch.zeros((1, block.filters)).cuda(cuda_rank)

                one_hot[:, weight_idx] = 1

                # if shift_size = 1, shift the current coordinate to its maximum and minimum obesrved value
                s_shift_down = one_hot * ((minima[sindex] - style_coords[sindex]) * shift_size)
                s_shift_up = one_hot * ((maxima[sindex] - style_coords[sindex]) * shift_size)

                for direction_index, shift in enumerate([s_shift_down, s_shift_up]):

                    shift = shift.squeeze(0)

                    current_style_layer.bias += shift
                    perturbed_generated_images = stylex.G(w_latent_tensor, noise)

                    shift_logits = classifier(perturbed_generated_images)
                    style_change_effect[0, direction_index, sindex] = shift_logits - image_generated_logits
            
                    current_style_layer.bias -= shift

            style_change_effects[image_index] = style_change_effect
            image_index += 1
        
        print("Initializing file")
        f = h5py.File(os.path.join(results_folder, 'style_change_records.hdf5'), 'w')
        file_style_change_effects = f.create_dataset('style_change', (num_images, 2, num_style_coords, 2), dtype='f')
        file_image_latents = f.create_dataset('latents', (num_images, 513), dtype='f')
        file_generated_image_classifications = f.create_dataset('base_prob', (num_images, 2), dtype='f')
        file_save_minima = f.create_dataset("minima", (1, num_style_coords), dtype='f')
        file_save_maxima = f.create_dataset("maxima", (1, num_style_coords), dtype='f')
        file_style_coordinates = f.create_dataset("style_coordinates", (num_images, num_style_coords), dtype='f')
        file_original_images = f.create_dataset("original_images", (num_images, 3, image_size, image_size), dtype='f')
        file_save_noise = f.create_dataset("noise", (1, image_size, image_size, 1), dtype='f')
        file_discriminator_results = f.create_dataset("discriminator", (num_images, 1), dtype='f')

        file_style_change_effects[:] = style_change_effects.cpu()
        file_image_latents[:] = image_latents.cpu()
        file_generated_image_classifications[:] = generated_image_classifications.cpu()
        file_style_coordinates[:] = style_coordinates.cpu()
        file_original_images[:] = original_images.cpu()
        file_discriminator_results[:] = discriminator_results.cpu()
        
        
        file_save_noise[0] = noise.cpu()
        file_save_minima[0] = minima.cpu()
        file_save_maxima[0] = maxima.cpu()
                
        f.close()
        
def filter_unstable_images(style_change_effect, effect_threshold = 0.3, num_indices_threshold = 150):

    unstable_images = (np.sum(np.abs(style_change_effect) > effect_threshold, axis=(1, 2, 3)) > num_indices_threshold)
    style_change_effect[unstable_images] = 0
    
    return style_change_effect


def find_significant_styles(style_change_effect,
                            num_indices, # number of significant styles
                            class_index,
                            generator,
                            classifier,
#                             all_dlatents,
#                             style_min,
#                             style_max,
                            max_image_effect = 0.2,
#                             label_size = 2,
                            sindex_offset = 0):
  
    num_images = style_change_effect.shape[0]

    # element-wise, resulting shape (num_images, num_dir*num_styles)
    style_effect_direction = np.maximum(0, style_change_effect[:, :, :, class_index].reshape((num_images, -1)))

    images_effect = np.zeros(num_images)
    all_sindices = []
    discriminator_removed = []

    while len(all_sindices) < num_indices:
        # find style index of images that haven't reached the maximum effect
        next_s = np.argmax(np.mean(style_effect_direction[images_effect < max_image_effect], axis=0))

        all_sindices.append(next_s)
        # accumulate shift on image 
        images_effect += style_effect_direction[:, next_s]
        # zero out the effects
        style_effect_direction[:, next_s] = 0

    return [(x // style_change_effect.shape[2], (x % style_change_effect.shape[2]) + sindex_offset) for x in all_sindices]


def generate_change_image_given_dlatent(dlatent,
                                        generator,
                                        classifier,
                                        class_index,
                                        sindex,
                                        s_style_min,
                                        s_style_max,
                                        style_direction_index,
                                        shift_size,
                                        label_size,
                                        noise, 
                                        cuda_rank):

    w_latent_tensor = styles_def_to_tensor(dlatent)
    
    image_generated, style_coords = generator(w_latent_tensor, noise, get_style_coords=True)

    block_idx, weight_idx = sindex_to_block_idx_and_index(generator, sindex)
    block = generator.blocks[block_idx]

    current_style_layer = None
    one_hot = None

    if weight_idx < block.input_channels:
        current_style_layer = block.to_style1
        one_hot = torch.zeros((1, block.input_channels)).cuda(cuda_rank)
    else:
        weight_idx -= block.input_channels
        current_style_layer = block.to_style2
        one_hot = torch.zeros((1, block.filters)).cuda(cuda_rank)

    one_hot[:, weight_idx] = 1

    if style_direction_index == 0:
        shift = one_hot * ((s_style_min - style_coords[:, sindex]) * shift_size).unsqueeze(1)
    else:
        shift = one_hot * ((s_style_max - style_coords[:, sindex]) * shift_size).unsqueeze(1)

    with torch.no_grad():
        shift = shift.squeeze(0)
        current_style_layer.bias += shift
        perturbed_generated_images, style_coords = generator(w_latent_tensor, noise, get_style_coords=True)
        shift_logits = classifier(perturbed_generated_images)
        change_prob = torch.softmax(shift_logits, dim=1).cpu().detach().numpy()[0, class_index]
        current_style_layer.bias -= shift

    return perturbed_generated_images, change_prob

def draw_on_image(image,
                  number,
                  font_file,
                  font_fill = (0, 0, 255)):

    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)
    fnt = ImageFont.truetype(font_file, 20)
    out_image = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(out_image)
    #draw.multiline_text((10, 10), ('%.3f' % number), font=fnt, fill=font_fill)
    return np.array(out_image)

def generate_images_given_dlatent(dlatent,
                                  generator,
                                  classifier,
                                  class_index,
                                  sindex,
                                  s_style_min,
                                  s_style_max,
                                  style_direction_index,
                                  font_file,
                                  noise,
                                  shift_size = 2,
                                  label_size = 2,
                                  draw_results_on_image = True,
                                  resolution = 64,
                                  cuda_rank = 0,
                                  gen_num_layers = 5):
  
    result_image = np.zeros((resolution, 2 * resolution, 3), np.uint8)
    dlatent = [(torch.Tensor(dlatent).cuda(cuda_rank), gen_num_layers)]
    w_latent_tensor = styles_def_to_tensor(dlatent).cuda(cuda_rank)

    base_image, style_coords = generator(w_latent_tensor, noise, get_style_coords=True)
    result = classifier(base_image)
    base_prob = torch.softmax(result, dim=1).cpu().detach().numpy()[0, class_index]

    if draw_results_on_image:
        result_image[:, :resolution, :] = draw_on_image(base_image[0].cpu().detach().numpy(), base_prob, font_file)
    else:
        result_image[:, :resolution, :] = (base_image[0].cpu().detach().numpy() * 127.5 + 127.5).astype(np.uint8)

    change_image, change_prob = (generate_change_image_given_dlatent(dlatent, generator, classifier,
                                                                     class_index, sindex,
                                                                     s_style_min, s_style_max,
                                                                     style_direction_index, shift_size,
                                                                     label_size, noise=noise, cuda_rank=cuda_rank))
 
    if draw_results_on_image:
        result_image[:, resolution:, :] = draw_on_image(change_image[0].cpu().detach().numpy(), change_prob, font_file)

    else:
        result_image[:, resolution:, :] = (np.maxiumum(np.minimum(change_image[0].cpu().detach().numpy(), 1), -1) * 127.5 + 127.5).astype(np.uint8)

    return (result_image, change_prob, base_prob)

def visualize_style(generator,
                    classifier,
                    all_dlatents,
                    style_change_effect,
                    style_min,
                    style_max,
                    sindex,
                    style_direction_index,
                    max_images,
                    shift_size,
                    font_file,
                    noise,
                    label_size = 2,
                    class_index = 0,
                    effect_threshold = 0.3,
                    seed = None,
                    allow_both_directions_change = False,
                    draw_results_on_image = True):
  
    if allow_both_directions_change:
        images_idx = (np.abs(style_change_effect[:, style_direction_index, sindex,
                                             class_index]) >
                  effect_threshold).nonzero()[0]
    else:
        images_idx = ((style_change_effect[:, style_direction_index, sindex,
                                       class_index]) >
                  effect_threshold).nonzero()[0]
    if images_idx.size == 0:
        return np.array([])

    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(images_idx)
    images_idx = images_idx[:min(max_images*10, len(images_idx))]
    dlatents = all_dlatents[images_idx]

    result_images = []
    for i in range(len(images_idx)):
        cur_dlatent = dlatents[i:i + 1]
        (result_image, base_prob, change_prob) = generate_images_given_dlatent(
                                                 dlatent=cur_dlatent,
                                                 generator=generator,
                                                 classifier=classifier,
                                                 class_index=class_index,
                                                 sindex=sindex,
                                                 noise=noise,
                                                 s_style_min=style_min[sindex],
                                                 s_style_max=style_max[sindex],
                                                 style_direction_index=style_direction_index,
                                                 font_file=font_file,
                                                 shift_size=shift_size,
                                                 label_size=label_size,
                                                 draw_results_on_image=draw_results_on_image,
                                                 gen_num_layers=stylex.G.num_layers)

        if np.abs(change_prob - base_prob) < effect_threshold:
            continue
        result_images.append(result_image)
        if len(result_images) == max_images:
            break

    if len(result_images) < 3:
        # No point in returning results with very little images
        return np.array([])
    return np.concatenate(result_images[:max_images], axis=0)


def visualize_style_by_distance_in_s(generator,
                                    classifier,
                                    all_dlatents,
                                    all_style_vectors_distances,
                                    style_min,
                                    style_max,
                                    sindex,
                                    style_sign_index,
                                    max_images,
                                    shift_size,
                                    font_file,
                                    noise,
                                    label_size = 2,
                                    class_index = 0,
                                    draw_results_on_image = True,
                                    effect_threshold = 0.1, 
                                    cuda_rank=0):

    images_idx = np.argsort(all_style_vectors_distances[:, sindex, style_sign_index])[::-1]

    if images_idx.size == 0:
        print("images_idx size is zero")
        return np.array([])

    images_idx = images_idx[:min(max_images*10, len(images_idx))]
    dlatents = all_dlatents[images_idx]

    result_images = []
    for i in range(len(images_idx)):
        cur_dlatent = dlatents[i:i + 1]
        (result_image, change_prob, base_prob) = generate_images_given_dlatent(dlatent=cur_dlatent,
                                                                                 generator=generator,
                                                                                 classifier=classifier,
                                                                                 class_index=class_index,
                                                                                 sindex=sindex,
                                                                                 noise=noise,
                                                                                 s_style_min=style_min[sindex],
                                                                                 s_style_max=style_max[sindex],
                                                                                 style_direction_index=style_sign_index,
                                                                                 font_file=font_file,
                                                                                 shift_size=shift_size,
                                                                                 label_size=label_size,
                                                                                 draw_results_on_image=draw_results_on_image,
                                                                                 cuda_rank = cuda_rank)
        result_images.append(result_image)

    if len(result_images) < 3:
        return np.array([])
    return np.concatenate(result_images[:max_images], axis=0)

def show_image(image, fmt='png'):

    if image.dtype == np.float32:
        image = np.uint8(image * 255)

    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(image)
    plt.show()
    
