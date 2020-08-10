import torch
from torch import optim
from torchvision import models

# Save a checkpoint
def save_checkpoint(classifier, arch, learning_rate, state_dict, class_to_idx, optimizer_dict, criterion):
    checkpoint = {'classifier': classifier,
                  'arch': arch,
                  'learning_rate': learning_rate,
                  'state_dict': state_dict,
                  'class_to_idx': class_to_idx,
                  'optimizer_dict': optimizer_dict,
                  'criterion': criterion}

    torch.save(checkpoint, 'checkpoint.pth')
    
    
# Load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    
    criterion = checkpoint['criterion']
    
    return optimizer, criterion, model