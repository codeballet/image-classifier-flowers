import torch


# Make code device agnostic with respect to user input
def get_device(gpu):
    if gpu == 'on':
        device = torch.device("cuda:0" if torch.cuda.is_available() 
                              else "cpu")
        
    else:
        device = torch.device("cpu")
        
    return device