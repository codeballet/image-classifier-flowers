from PIL import Image
import torch
import numpy as np


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an torch tensor
    '''
    
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    # Crop a square from the image
    im_width, im_height = im.size
    crop = min(im.size)
    im_square = im.crop(((im_width - crop) // 2,
                         (im_height - crop) // 2,
                         (im_width + crop) // 2,
                         (im_height + crop) // 2))
    
    # Create thumbnail picture
    thumb_size = (256, 256)
    im_square.thumbnail(thumb_size)
    
    # Crop thumbnail
    thumb_crop = 224
    left = int(im_square.size[0]/2 - thumb_crop/2)
    upper = int(im_square.size[1]/2 - thumb_crop/2)
    right = left + thumb_crop
    lower = upper + thumb_crop
    im_thumb_crop = im_square.crop((left, upper, right, lower))
    
    # Convert image to np array and normalize color channels
    np_image = np.array(im_thumb_crop)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose the color channel to be the first dimension for PyTorch
    # instead of the third dimension as in the PIL image
    np_image = np_image.transpose(2, 0, 1)
    
    return torch.from_numpy(np_image)