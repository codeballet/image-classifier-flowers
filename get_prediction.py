import torch

from get_parser_predict import args
from get_device import get_device
from get_names import cat_to_name
from get_process_image import process_image


def predict(image_path, model, cat_names, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Get device
    device = get_device(args.gpu)
    
    # Predict the class from an image file
    image = process_image(image_path).unsqueeze_(0).type(torch.FloatTensor)
    image = image.to(device)

    # Run the model
    with torch.no_grad():
        model.to(device)
        model.eval()
        logps = model.forward(image)

    # Get the top probabilities and classes
    ps = torch.exp(logps)
    top_ps, top_idx = ps.topk(topk, dim=1)
    
    # Convert tensors to lists
    probabilities = top_ps.tolist()[0]
    indeces = top_idx.tolist()[0]
        
    # Invert the dictionary class_to_idx
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # Create a list of classes from the indeces
    labels = [idx_to_class[idx] for idx in indeces]
    
    # Create a list of flowers from classes
    names_dic = cat_to_name(cat_names)
    flowers = [names_dic[label] for label in labels]
    
    return probabilities, labels, flowers